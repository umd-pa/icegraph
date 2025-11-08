# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Union, Optional, Self, Type, Dict, TYPE_CHECKING
from pathlib import Path

import torch
import torch.optim as optim
from torch.optim import Optimizer
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import LRScheduler
from torch_geometric.seed import seed_everything
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP  # DDP

from icegraph.data import DatasetRegistry
from icegraph.types import ComputedMetrics
from icegraph.config import IGConfig
from icegraph.trainer.core._config import TrainerConfig
from icegraph.trainer.arch import ModelFactory
from icegraph.utils.pathutils import PathResolver
from icegraph.trainer.callbacks import ConsoleCallback, ExportCallback, TensorBoardCallback, Callback
from icegraph.trainer.callbacks.base import CallbackRegistryMixin
from icegraph.trainer.normalizers import resolve_normalizer, Normalizer
from icegraph.trainer.base.exceptions import EmptyDataLoaderError
from icegraph.trainer.interfaces import resolve_strategy
from icegraph.trainer.interfaces.base import TaskStrategy

if TYPE_CHECKING:
    from icegraph.trainer.distributed import DDPProcessState

__all__ = ["Trainer"]

_RANK0_ONLY_HOOKS = {
    "on_execute",
    "on_epoch_begin",
    "on_epoch_end",
    "on_validation_begin",
    "on_validation_end",
    "on_test_begin",
    "on_test_end",
    "on_save"
}


class Trainer(CallbackRegistryMixin, torch.nn.Module):
    """
    Trainer class for managing the training, validation, and testing of a PyTorch model
    using datasets registered in a `DatasetRegistry`.

    This class handles model optimization, loss computation, and RMSE reporting
    for each stage of the training lifecycle.
    """

    def __init__(
        self,
        registry: DatasetRegistry, *,
        outdir: Optional[Union[str, Path]] = None,
        process_state: Optional[DDPProcessState] = None,
        debug: bool = False,
        default_callbacks: bool = True
    ) -> None:
        """
        Initialize the Trainer.

        Sets up global configuration, reproducibility, datasets, model, optimizer,
        loss function, and metrics tracking.

        Args:
            registry (DatasetRegistry): Dataset registry containing dataloaders.
            outdir (Optional[Union[str, Path]]): Path to save the trained model and any other generated files.
            process_state (Optional[DDPProcessState]): Utility object for distributed jobs.
            debug (bool): Enable debug mode.
            default_callbacks (bool): Whether to register default callbacks. Defaults to True.
        """
        # call to super
        super().__init__()

        # set debug flag
        self._debug = debug

        # store the registry object
        self.registry = registry

        # grab the global config
        self._config = IGConfig.get()
        # build a trainer config object
        self.trainer_config = TrainerConfig.from_config(self._config)

        # initialize the DDP process state handler
        # this needs to happen before any other initialization as downstream relies on process state
        self._process_state = self._init_ddp(process_state)

        # init device
        self.device = self._init_device()

        # initialize the strategy and grab the loss func
        # this needs to happen before model init as it depends on strategy
        self.strategy = self._init_strategy()
        self.loss_fn = self.strategy.loss_function()

        # set up the model
        self.model = self._init_model()

        # get the output directory
        self.outdir = self._init_output_dir(outdir)
        # place logs alongside run files
        self.log_dir = self.outdir / "logs"

        # set global seed for reproducibility
        self._init_seed()

        # register callbacks
        self._init_default_callbacks()

        # initialize the console
        self.console = self._init_console()

        # initialize the normalizer
        self.normalizer = self._init_normalizer()

        # initialize the optimizer
        self.optimizer = self._init_optimizer()

        # initialize the scheduler
        self.scheduler = self._init_scheduler()

        # init metric dicts
        self._train_metrics:    Dict[int, ComputedMetrics] = {}
        self._val_metrics:      Dict[int, ComputedMetrics] = {}
        self._test_metrics:     Dict[int, ComputedMetrics] = {}

        # setup eval stash dict
        self._last_eval: Dict[str, Dict[str, Optional[torch.Tensor]]] = {
            "val": {"preds": None, "targets": None, "includes": None},
            "test": {"preds": None, "targets": None, "includes": None},
        }

        # global access to current epoch
        self.current_epoch: int

        # calculate per-split batch counts
        self.train_batch_count: int     = len(self.registry.train_dataloader)
        self.val_batch_count:   int     = len(self.registry.val_dataloader)
        self.test_batch_count:  int     = len(self.registry.test_dataloader)

        # run initialization for each registered callback
        self._gated_fire("on_init")

    ### INIT METHODS

    @staticmethod
    def _init_ddp(process_state: Optional[DDPProcessState]) -> Optional[DDPProcessState]:
        """Initialize the DDP process state handler if there is one."""
        if process_state is not None:
            process_state.init()

        return process_state

    def _init_device(self) -> torch.device:
        """Select and initialize the device."""
        # get the device selection
        if self._process_state and torch.cuda.is_available():
            device = torch.device(f"cuda:{self._process_state.procinfo['local_rank']}")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        return device

    def _init_model(self) -> torch.nn.Module:
        """Initialize the model for training."""
        # determine dimensions of input and output
        in_channels = self.strategy.in_channels(self)
        out_channels = self.strategy.out_channels(self)

        # pull hidden layers and channels from config (not controlled by strategy)
        hidden_channels = self.trainer_config.hidden_channels
        hidden_layers = self.trainer_config.hidden_layers

        # get the model and move to accelerator
        active_model = ModelFactory.create(
            "gravnet", in_channels, hidden_channels, out_channels, hidden_layers
        )
        model = active_model.to(self.device)

        # if ddp is enabled, the model needs to be wrapped by torch DDP
        if self._process_state:
            model = DDP(
                model,
                device_ids=[self.device.index] if self.device.type == "cuda" else None,
                output_device=(self.device.index if self.device.type == "cuda" else None),
                broadcast_buffers=False,
                gradient_as_bucket_view=True,
                find_unused_parameters=False
            )

        return model

    @staticmethod
    def _init_output_dir(outdir: Optional[Union[str, Path]]) -> Path:
        """Resolve the output directory."""
        # resolve the output path
        resolver = PathResolver(path=outdir, origin=None, extension=None, stage="trainer")
        return resolver.resolve(return_dir=True)

    def _init_seed(self) -> None:
        """Set a deterministic seed."""
        base_seed = self.trainer_config.seed
        rank_off = (self._process_state.procinfo["rank"] if self._process_state else 0)
        self._set_seed(base_seed + rank_off)

    def _init_strategy(self) -> TaskStrategy:
        """Initialize the trainer strategy."""
        strategy_selection = self._config.user_config.training.strategy.task
        strategy_kwargs = self._config.user_config.training.strategy.kwargs.toDict()

        strategy_spec = resolve_strategy(strategy_selection, call=False)
        return strategy_spec(**strategy_kwargs)

    def _init_default_callbacks(self) -> None:
        """Set up minimal default callbacks."""
        default_callbacks = [
            ExportCallback(),
            TensorBoardCallback()
        ]

        for cb in default_callbacks:
            self.register_callback(cb)

    def _init_console(self) -> Callback:
        """Initialize the console callback."""
        console = ConsoleCallback()
        self.register_callback(console)

        return console

    def _init_normalizer(self) -> Normalizer:
        """Initialize the normalizer."""
        # grab normalizer
        norm_selection = self._config.user_config.training.normalizer
        normalizer = resolve_normalizer(norm_selection)
        self.register_callback(normalizer)

        return normalizer

    def _init_optimizer(self) -> Optimizer:
        """Initialize the optimizer."""
        optimizer_str: str = self.trainer_config.optimizer
        if not hasattr(optim, optimizer_str):
            raise ValueError(f"Optimizer {optimizer_str} not found in torch.optim")

        optimizer_spec: Type[Optimizer] = getattr(optim, optimizer_str)
        optimizer = optimizer_spec(
            self.model.parameters(),
            **self.trainer_config.optimizer_kwargs
        )

        return optimizer

    def _init_scheduler(self) -> Optional[LRScheduler]:
        """Initialize the learning rate scheduler."""
        scheduler_step_mode = self.trainer_config.scheduler_step_mode
        scheduler_str: Optional[str] = self.trainer_config.scheduler
        if scheduler_str is not None:
            if not hasattr(lr_scheduler, scheduler_str):
                raise ValueError(f"LRScheduler '{scheduler_str}' not found in torch.optim.lr_scheduler")

            scheduler_spec: Type[LRScheduler] = getattr(lr_scheduler, scheduler_str)
            scheduler = scheduler_spec(
                self.optimizer,
                **self.trainer_config.scheduler_kwargs
            )
        else:
            scheduler = None

        if scheduler:
            scheduler.step_mode = scheduler_step_mode

        return scheduler

    ### ---

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.exit(exc_type is KeyboardInterrupt)
        return False

    def _gated_fire(self, hook_name: str, *args, **kwargs) -> None:
        """
        Invokes internal callback hooks but first gates by rank.

        Args:
            hook_name (str): The name of the callback method to call.
            *args: Positional arguments to forward into the callback.
            **kwargs: Keyword arguments to forward into the callback.
        """
        rank0_only = hook_name in _RANK0_ONLY_HOOKS

        if rank0_only and self._process_state and not self._process_state.is_main_process():
            return

        self._fire(hook_name, *args, **kwargs)

    @property
    def last_eval(self) -> Dict[str, Dict[str, Optional[torch.Tensor]]]:
        """Getter for the last eval stash dict."""
        return self._last_eval

    @staticmethod
    def _set_seed(seed: int) -> None:
        """
        Configure seeds and deterministic flags for full reproducibility.

        Args:
            seed (int): Seed for torch, NumPy, and Python random.
        """
        # seed everything
        seed_everything(seed)

        # ensure deterministic setup
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _forward_and_target(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a forward pass and align targets to batch-aggregated outputs.

        Args:
            batch (Batch): A PyG batch.

        Returns:
            out (Tensor): Model predictions per graph in the batch.
            target (Tensor): True values aggregated by graph, shaped to match `out`.
        """
        out = self.model(batch.x, batch.batch)  # type: ignore[arg-type]
        target = self.strategy.adapt_targets(batch, out)

        return out, target

    def _train_batchwise(self, dataloader: DataLoader) -> ComputedMetrics:
        """
        Run one epoch of training over `dataloader` and collect SSE and sample counts.

        Args:
            dataloader (DataLoader): Yields batches for training.

        Returns:
            ComputedMetrics: Contains metrics for the epoch.
        """
        metrics = self.strategy.make_metrics()

        # make sure correct mode is active
        self.model.train()

        # iterate over each batch
        for idx, batch in enumerate(dataloader):
            self._gated_fire("on_batch_begin", batch)

            batch = batch.to(self.device, non_blocking=True)
            self._gated_fire("on_batch_transfer", batch)

            out, target = self._forward_and_target(batch)
            loss = self.loss_fn(out, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # step the scheduler if there is one
            if self.scheduler is not None:
                if self.scheduler.step_mode == "batch":
                    self.scheduler.step()
                if self.scheduler.step_mode == "warm_restarts":
                    # epoch + progress-in-epoch (PyTorch docs recommend this pattern)
                    t_cur = self.current_epoch + (idx + 1) / self.train_batch_count
                    self.scheduler.step(t_cur)

            # task-agnostic accumulation
            metrics.update(out.detach(), target.detach(), loss.detach())

            self._gated_fire("on_batch_end", batch, out.detach(), target.detach(), loss.detach(), metrics)

        if self.scheduler is not None and self.scheduler.step_mode == "epoch":
            self.scheduler.step()

        return metrics.compute()

    def _evaluate_batchwise(self, dataloader: DataLoader, *, stash: Optional[str] = None) -> ComputedMetrics:
        """
        Run one pass of evaluation (no gradient) over `dataloader`.

        Args:
            dataloader (DataLoader): Yields batches for validation or testing.
            stash (Optional[str]): Whether to stash results.

        Returns:
            ComputedMetrics: Contains metrics for the run.
        """
        metrics = self.strategy.make_metrics()

        # make sure correct mode is active
        self.model.eval()

        # stashing
        collect = stash in {"val", "test"}
        outs: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        includes: list[torch.Tensor] = []

        # use no_grad on eval loops
        with torch.inference_mode():
            # iterate over each batch
            for batch in dataloader:
                self._gated_fire("on_batch_begin", batch)

                batch = batch.to(self.device, non_blocking=True)
                self._gated_fire("on_batch_transfer", batch)

                out, target = self._forward_and_target(batch)
                out, target, mask = self.strategy.filter_eval(out, target)
                loss = self.loss_fn(out, target)

                metrics.update(out.detach(), target.detach(), loss.detach(), mask=mask)

                self._gated_fire("on_batch_end", batch, out.detach(), target.detach(), loss.detach(), metrics)

                if collect:
                    outs.append(out.detach().to("cpu", non_blocking=True))
                    targets.append(target.detach().to("cpu", non_blocking=True))

                    # grab includes if present
                    if hasattr(batch, "include_labels"):
                        includes.append(batch.include_labels.detach().to("cpu", non_blocking=True))

            if collect:
                self._last_eval[stash]["preds"]     = torch.cat(outs, dim=0) if outs else None
                self._last_eval[stash]["targets"]   = torch.cat(targets, dim=0) if targets else None
                self._last_eval[stash]["includes"]  = torch.cat(includes, dim=0) if includes else None

        return metrics.compute()

    @property
    def train_metrics(self) -> dict[int, ComputedMetrics]:
        """Getter for training metrics."""
        return self._train_metrics

    @property
    def val_metrics(self) -> dict[int, ComputedMetrics]:
        """Getter for validation metrics."""
        return self._val_metrics

    @property
    def test_metrics(self) -> dict[int, ComputedMetrics]:
        """Getter for testing metrics."""
        return self._test_metrics

    def _train(self, epoch: int) -> None:
        """
        Run a single training epoch.
        """
        self._gated_fire("on_train_begin", epoch)
        metrics_dict = self._train_batchwise(self.registry.train_dataloader)

        if not metrics_dict or all(v != v for v in metrics_dict.values()):  # all NaN
            raise EmptyDataLoaderError("No data in dataloader; cannot train/validate.")

        self._train_metrics[self.current_epoch + 1] = metrics_dict

        self._gated_fire("on_train_end", epoch, metrics_dict)

    def _validate(self, epoch: int) -> None:
        """
        Compute validation metrics without altering model weights.

        Args:
            epoch (int): The epoch index for logging/scalar steps.
        """
        self._gated_fire("on_validation_begin", epoch)
        metrics_dict = self._evaluate_batchwise(self.registry.val_dataloader, stash="val")

        if not metrics_dict or all(v != v for v in metrics_dict.values()):
            raise EmptyDataLoaderError("No data in dataloader; cannot train/validate.")

        self._val_metrics[epoch + 1] = metrics_dict

        self._gated_fire("on_validation_end", epoch, metrics_dict)

    def _test(self, epoch: int) -> None:
        """
        Compute test metrics using the final model (no weight updates).

        Args:
            epoch (int): The epoch index for logging/scalar steps.
        """
        self._gated_fire("on_test_begin", epoch)
        metrics_dict = self._evaluate_batchwise(self.registry.test_dataloader, stash="test")

        if not metrics_dict or all(v != v for v in metrics_dict.values()):
            raise EmptyDataLoaderError("No data in dataloader; cannot train/validate.")

        self._test_metrics[epoch + 1] = metrics_dict

        self._gated_fire("on_test_end", epoch, metrics_dict)

    def save(self, epoch: Optional[int] = None, metrics: Optional[ComputedMetrics] = None) -> None:
        """
        Saves the model if there is an associated callback.
        """
        self._gated_fire("on_save", epoch, metrics)


    def execute(self) -> None:
        """
        Method for executing the full pipeline: training, validation at set intervals, final testing, and teardown.
        """
        # fire on_execute callback hook
        self._gated_fire("on_execute")

        # iterate over epochs
        # self.current_epoch is always set before any usage, so not checking via hasattr is fine here
        for self.current_epoch in range(self.trainer_config.max_epochs):
            # sampler epoch
            sampler = getattr(self.registry.train_dataloader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(self.current_epoch)

            self._gated_fire("on_epoch_begin", self.current_epoch)

            # train on all ranks
            self._train(epoch=self.current_epoch)

            # validate only on main rank
            val_interval = self.trainer_config.val_interval
            if (not self._process_state or self._process_state.is_main_process()) and val_interval > 0 and (
                    self.current_epoch + 1) % val_interval == 0:
                self._validate(epoch=self.current_epoch)

            # test only once at end, on main rank
            if (not self._process_state or self._process_state.is_main_process()) and (self.current_epoch + 1) == self.trainer_config.max_epochs:
                self._test(epoch=self.current_epoch)

            self._gated_fire("on_epoch_end", self.current_epoch)

            self._process_state.barrier()  # keep ranks in lockstep (no-op if not DDP)

    def exit(self, interrupt: bool) -> None:
        try:
            self._close_resources()
            self._gated_fire("on_teardown")
        finally:
            if self._process_state:
                self._process_state.cleanup(interrupt)

    def _close_resources(self) -> None:
        """Attempt to close all resources before shutdown."""
        self.registry.close()
