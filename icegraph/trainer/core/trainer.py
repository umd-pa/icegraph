# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import Union, Optional, List, Self, Type, Dict
from pathlib import Path
import math

import torch
import torch.optim as optim
from torch.optim import Optimizer
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import LRScheduler
from torch_geometric.seed import seed_everything
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP  # DDP

from icegraph.trainer.distributed import ddp
from icegraph.data import DatasetRegistry
from icegraph.types import ComputedMetrics
from icegraph.config import IGConfig
from icegraph.trainer.core._config import TrainerConfig
from icegraph.trainer.arch import ModelFactory
from icegraph.utils.pathutils import PathResolver
from icegraph.trainer.callbacks import ConsoleCallback, ExportCallback, TensorBoardCallback, Callback
from icegraph.trainer.normalizers import resolve_normalizer, Normalizer
from icegraph.trainer.base.exceptions import EmptyDataLoaderError, TrainerError
from icegraph.trainer.interfaces import resolve_strategy
from icegraph.trainer.interfaces.base import TaskStrategy

__all__ = ["Trainer"]

_RANK0_ONLY_HOOKS = {
    "on_execute",
    "on_epoch_begin",
    "on_epoch_end",
    "on_validation_begin",
    "on_validation_end",
    "on_test_begin",
    "on_test_end",
    "on_save",
}


class Trainer(torch.nn.Module):
    """
    Trainer class for managing the training, validation, and testing of a PyTorch model
    using datasets registered in a `DatasetRegistry`.

    This class handles model optimization, loss computation, and RMSE reporting
    for each stage of the training lifecycle.
    """

    def __init__(
        self,
        dataset_registry: DatasetRegistry, *,
        callbacks: Optional[list[Callback]] = None,
        trainer_config: Optional[TrainerConfig] = None,
        outdir: Optional[Union[str, Path]] = None,
        model: str = "gravnet",
        device: str = "cuda",
        normalizer: Optional[Normalizer] = None
    ) -> None:
        """
        Initialize the Trainer.

        Sets up global configuration, reproducibility, datasets, model, optimizer,
        loss function, and metrics tracking.

        Args:
            dataset_registry (DatasetRegistry): Dataset registry containing dataloaders.
            callbacks (Optional[list[Callback]]): List of callbacks to pass to the trainer.
            trainer_config (TrainerConfig): A TrainerConfig instance with training params.
            outdir (Optional[Union[str, Path]]): Path to save the trained model and any other generated files.
            model (str): The model to be trained and evaluated, must be one of ['gravnet'].
            device (str): Preferred device for computation. Defaults to 'cuda'.
            normalizer (Optional[normalizer]): Normalizer to use, overrides any normalizer specified in config.
        """
        super().__init__()

        self._dist = ddp.init("nccl")

        self._config = IGConfig.get()
        self.trainer_config = trainer_config or TrainerConfig.from_config(self._config)

        # resolve the output path
        resolver = PathResolver(path=outdir, origin=None, extension=None, stage="trainer")
        self.outdir = resolver.resolve(return_dir=True)

        # place log dir next to run files
        self.log_dir = self.outdir / "logs"

        # set global seed for reproducibility
        base_seed = self.trainer_config.seed
        rank_off = (self._dist["rank"] if self._dist else 0)
        self._set_seed(base_seed + rank_off)

        # load datasets and device
        self.registry = dataset_registry

        # load strategy
        strategy_selection = self._config.user_config.training.strategy.task
        self.strategy_kwargs = self._config.user_config.training.strategy.kwargs.toDict()

        strategy_spec = resolve_strategy(strategy_selection, call=False)
        self.strategy: TaskStrategy = strategy_spec(**self.strategy_kwargs)

        # get the device selection
        if self._dist and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self._dist['local_rank']}")
        else:
            self.device = torch.device("cuda" if (torch.cuda.is_available() and device == "cuda") else "cpu")

        self.console = ConsoleCallback()

        # grab callbacks
        default_callbacks = [
            ExportCallback(),
            TensorBoardCallback()
        ]
        self.callbacks = [*callbacks, self.console] if callbacks else [*default_callbacks, self.console]

        # grab normalizer
        # make sure the user didnt pass any normalizers in callbacks
        norm_selection = self._config.user_config.training.normalizer
        self.normalizer = normalizer or resolve_normalizer(norm_selection)
        self._ensure_single_normalizer()

        # determine dimensions of input and output
        in_channels = self.strategy.in_channels(self)
        out_channels = self.strategy.out_channels(self)

        # get the model
        active_model = ModelFactory.create(
            model, in_channels, self.trainer_config.hidden_channels, out_channels, self.trainer_config.hidden_layers
        )
        self.model = active_model.to(self.device)

        if self._dist:
            self.model = DDP(
                self.model,
                device_ids=[self.device.index] if self.device.type == "cuda" else None,
                output_device=(self.device.index if self.device.type == "cuda" else None),
                broadcast_buffers=False,
                gradient_as_bucket_view=True,
                find_unused_parameters=False,  # set True only if you truly skip params in forward
            )

        # initialize the optimizer
        optimizer_str: str = self.trainer_config.optimizer
        if not hasattr(optim, optimizer_str):
            raise ValueError(f"Optimizer {optimizer_str} not found in torch.optim")

        optimizer: Type[Optimizer] = getattr(optim, optimizer_str)
        self.optimizer = optimizer(
            active_model.parameters(),
            **self.trainer_config.optimizer_kwargs
        )

        self.scheduler: Optional[LRScheduler]
        self.scheduler_step_mode = self.trainer_config.scheduler_step_mode
        scheduler_str: Optional[str] = self.trainer_config.scheduler
        if scheduler_str is not None:
            if not hasattr(lr_scheduler, scheduler_str):
                raise ValueError(f"LRScheduler {scheduler_str} not found in torch.optim.lr_scheduler")

            scheduler: Type[LRScheduler] = getattr(lr_scheduler, scheduler_str)
            self.scheduler = scheduler(
                self.optimizer,
                **self.trainer_config.scheduler_kwargs
            )
        else:
            self.scheduler = None

        # loss function
        self.loss_fn = self.strategy.loss_function()
        self.strategy.post_init_check(self.model)  # run post init check if any

        # init metric dicts
        self._train_metrics: Dict[int, ComputedMetrics] = {}
        self._val_metrics: Dict[int, ComputedMetrics] = {}
        self._test_metrics: Dict[int, ComputedMetrics] = {}

        # setup stash dict
        self._last_eval: Dict[str, Dict[str, Optional[torch.Tensor]]] = {
            "val": {"preds": None, "targets": None, "includes": None},
            "test": {"preds": None, "targets": None, "includes": None},
        }

        # global access to current epoch
        self.current_epoch: Optional[int] = None

        # calculate per-split batch counts
        batch_size = self._config.user_config.training.batch_size
        self.train_batch_count = math.ceil(len(self.registry.train_dataset) / batch_size)
        self.val_batch_count = math.ceil(len(self.registry.val_dataset) / batch_size)
        self.test_batch_count = math.ceil(len(self.registry.test_dataset) / batch_size)

        self._fire("on_init")

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        try:
            self._close_resources()
        finally:
            self._fire("on_teardown")
            if self._dist:
                ddp.cleanup()

    def _close_resources(self) -> None:
        """Close dataset resources."""
        for attr in ("train_dataset", "val_dataset", "test_dataset"):
            dataset = getattr(self.registry, attr, None)
            if dataset is not None and hasattr(dataset, "close"):
                try:
                    dataset.close()
                except Exception:
                    pass

        for attr in ("train_dataloader", "val_dataloader", "test_dataloader"):
            loader = getattr(self.registry, attr, None)
            if loader is None:
                continue
            if getattr(loader, "persistent_workers", False):
                loader_iterator = getattr(loader, "_iterator", None)
                if loader_iterator is not None:
                    try:
                        # attempt to directly shutdown workers
                        loader_iterator._shutdown_workers()
                    except Exception:
                        pass

    @property
    def last_eval(self) -> Dict[str, Dict[str, Optional[torch.Tensor]]]:
        """Getter for the last eval stash dict."""
        return self._last_eval

    def register_callback(self, callback: Callback) -> None:
        """Register a callback."""
        if not isinstance(callback, Callback):
            raise TypeError("callback must be a subclass of 'Callback'")
        self.callbacks.append(callback)

        # make sure user didnt pass a normalizer callback
        self._ensure_single_normalizer()

        # need to initialize this new callback as it wasn't caught in trainer's __init__
        self.callbacks[-1].on_init(self)

    def _ensure_single_normalizer(self) -> None:
        _normalizers: List[Normalizer] = [cb for cb in self.callbacks + [self.normalizer] if isinstance(cb, Normalizer)]

        if len(_normalizers) == 0:
            raise TrainerError(
                "No normalizer found. Exactly one normalizer is required, which must be an instance of 'Normalizer'."
            )
        if len(_normalizers) > 1:
            names = ", ".join(type(cb).__name__ for cb in _normalizers)
            raise TrainerError(f"Multiple normalizers found ({names}). Exactly one normalizer is allowed.")

    def _fire(self, hook_name: str, *args, **kwargs) -> None:
        """
        Invoke a hook on every registered callback.

        Args:
            hook_name (str): The name of the callback method to call.
            *args: Positional arguments to forward into the callback.
            **kwargs: Keyword arguments to forward into the callback.
        """
        rank0_only = hook_name in _RANK0_ONLY_HOOKS
        if rank0_only and self._dist and not ddp.is_main_process():
            return
        for cb in (*self.callbacks, self.normalizer):
            getattr(cb, hook_name)(self, *args, **kwargs)

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
            self._fire("on_batch_begin", batch)

            batch = batch.to(self.device, non_blocking=True)
            self._fire("on_batch_transfer", batch)

            out, target = self._forward_and_target(batch)
            loss = self.loss_fn(out, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # step the scheduler if there is one
            if self.scheduler is not None and self.scheduler_step_mode == "batch":
                self.scheduler.step()
            if self.scheduler is not None and self.scheduler_step_mode == "warm_restarts":
                # epoch + progress-in-epoch (PyTorch docs recommend this pattern)
                t_cur = self.current_epoch + (idx + 1) / self.batch_count
                self.scheduler.step(t_cur)

            # task-agnostic accumulation
            metrics.update(out.detach(), target.detach(), loss.detach())

            self._fire("on_batch_end", batch, out.detach(), target.detach(), loss.detach(), metrics)

        if self.scheduler is not None and self.scheduler_step_mode == "epoch":
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
                self._fire("on_batch_begin", batch)

                batch = batch.to(self.device, non_blocking=True)
                self._fire("on_batch_transfer", batch)

                out, target = self._forward_and_target(batch)
                out, target, mask = self.strategy.filter_eval(out, target)
                loss = self.loss_fn(out, target)

                metrics.update(out.detach(), target.detach(), loss.detach(), mask=mask)

                self._fire("on_batch_end", batch, out.detach(), target.detach(), loss.detach(), metrics)

                if collect:
                    outs.append(out.detach().to("cpu", non_blocking=True))
                    targets.append(target.detach().to("cpu", non_blocking=True))

                    # grab includes if present
                    if hasattr(batch, "include_labels"):
                        includes.append(batch.include_labels.detach().to("cpu", non_blocking=True))

            if collect:
                self._last_eval[stash]["preds"] = torch.cat(outs, dim=0) if outs else None
                self._last_eval[stash]["targets"] = torch.cat(targets, dim=0) if targets else None
                self._last_eval[stash]["includes"] = torch.cat(includes, dim=0) if includes else None

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
        self._fire("on_train_begin", epoch)
        metrics_dict = self._train_batchwise(self.registry.train_dataloader)

        if not metrics_dict or all(v != v for v in metrics_dict.values()):  # all NaN
            raise EmptyDataLoaderError("No data in dataloader; cannot train/validate.")

        self._train_metrics[self._current_epoch + 1] = metrics_dict

        self._fire("on_train_end", epoch, metrics_dict)

    def _validate(self, epoch: int) -> None:
        """
        Compute validation metrics without altering model weights.

        Args:
            epoch (int): The epoch index for logging/scalar steps.
        """
        self._fire("on_validation_begin", epoch)
        metrics_dict = self._evaluate_batchwise(self.registry.val_dataloader, stash="val")

        if not metrics_dict or all(v != v for v in metrics_dict.values()):
            raise EmptyDataLoaderError("No data in dataloader; cannot train/validate.")

        self._val_metrics[epoch + 1] = metrics_dict

        self._fire("on_validation_end", epoch, metrics_dict)

    def _test(self, epoch: int) -> None:
        """
        Compute test metrics using the final model (no weight updates).

        Args:
            epoch (int): The epoch index for logging/scalar steps.
        """
        self._fire("on_test_begin", epoch)
        metrics_dict = self._evaluate_batchwise(self.registry.test_dataloader, stash="test")

        if not metrics_dict or all(v != v for v in metrics_dict.values()):
            raise EmptyDataLoaderError("No data in dataloader; cannot train/validate.")

        self._test_metrics[epoch + 1] = metrics_dict

        self._fire("on_test_end", epoch, metrics_dict)

    def save(self, epoch: Optional[int] = None, metrics: Optional[ComputedMetrics] = None) -> None:
        """
        Saves the model if there is an associated callback.
        """
        self._fire("on_save", epoch, metrics)

    def execute(self) -> None:
        """
        Execute the full pipeline: training, validation at set intervals, final testing, and teardown.
        """
        # fire on_execute callback hook
        self._fire("on_execute")

        # iterate over epochs
        for self._current_epoch in range(self.trainer_config.max_epochs):
            # sampler epoch
            if self._dist and getattr(self, "_train_sampler", None) is not None:
                self._train_sampler.set_epoch(self._current_epoch)
            else:
                sampler = getattr(self.registry, "sampler", None)
                if sampler is not None:
                    sampler.set_epoch(self._current_epoch)

            self._fire("on_epoch_begin", self._current_epoch)

            # train on all ranks
            self._train(epoch=self._current_epoch)

            # validate only on main rank
            val_interval = self.trainer_config.val_interval
            if (not self._dist or ddp.is_main_process()) and val_interval > 0 and (
                    self._current_epoch + 1) % val_interval == 0:
                self._validate(epoch=self._current_epoch)

            # test only once at end, on main rank
            if (not self._dist or ddp.is_main_process()) and (self._current_epoch + 1) == self.trainer_config.max_epochs:
                self._test(epoch=self._current_epoch)

            self._fire("on_epoch_end", self._current_epoch)

            ddp.barrier()  # keep ranks in lockstep (no-op if not DDP)
