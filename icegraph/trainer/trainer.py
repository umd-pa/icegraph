# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Self, Type, TYPE_CHECKING
from pathlib import Path
from contextlib import nullcontext
import time

# torch imports
import torch
from torch import Tensor
import torch.optim as optim
from torch.optim import Optimizer
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import LRScheduler
from torch_geometric.seed import seed_everything
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from torch.nn.parallel import DistributedDataParallel  # DDP

# local imports
from icegraph.config import IGConfig
from .callbacks import context
from icegraph.trainer.model import ModelFactory
from icegraph.trainer.callbacks import ConsoleCallback, ExportCallback, TensorBoardCallback
from icegraph.trainer.callbacks import CallbackRegistry
from icegraph.trainer.metric import MetricFactory

from .types import Params

# components
from .components.normalizers import Normalizer, NormalizerFactory, NormalizerContext
from .components.normalizers.types import StatSurface

from icegraph.trainer.services.strategy import Strategy, StrategyFactory, StrategyContext
from icegraph.trainer.services.strategy.types import AttributeSurface

# systems
from .services.data import DataService, DatasetContext

# types
from icegraph.types.data import Split, ModelInputRole
from icegraph.types.files import SourceType, Source

if TYPE_CHECKING:
    from icegraph.trainer.callbacks.base import Callback
    from icegraph.trainer.metric import MetricRegistry
    from icegraph.trainer.dist import DDPProcessState

__all__ = ["Trainer"]


# module logger
import logging
logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer class responsible for managing the training, validation, and testing
    lifecycle of a PyTorch model.

    This class is the central coordinator for all training processes, including
    model execution, optimization, metrics, and any other callbacks.
    """

    def __init__(
        self,
        source: Source | SourceType,
        outdir: str | Path, *,
        process_state: DDPProcessState | None = None,
        default_callbacks: bool = True
    ) -> None:
        """
        Initialize the Trainer.

        Sets up global configuration, reproducibility, datasets, model, optimizer,
        loss function, and metrics tracking.

        Args:
            source (SourceType | Source): Source files to train from.
            outdir (str | Path): Path to save the trained model and any other generated files.
            process_state (DDPProcessState | None): Utility object for distributed jobs.
            default_callbacks (bool): Whether to register default callbacks. Defaults to True.
        """
        # call to super
        super().__init__()

        logger.debug("initializing %s", self.__class__.__name__)

        # stash output path
        self.outdir = Path(outdir)

        # build a trainer config dotmap
        self.config = IGConfig.get().user_config.training

        # global access to current epoch
        self.current_epoch: int = 0

        # slots for active mode and split
        self.split: Split = Split.TRAIN  # first split is always train

        # initialize the DDP process state handler
        # this needs to happen before any other initialization as downstream relies on process state
        self.process_state = self._init_ddp(process_state)

        # initialize the dataset manager
        self.data = self._init_data(source)

        # initialize the callback registry before initializing any callbacks
        self.callbacks = self._init_callback_registry()

        # init device
        self.device = self._init_device()

        # initialize the strategy and grab the loss func
        # this needs to happen before model init as it depends on strategy
        self.strategy: Strategy = self._init_strategy()
        self.loss_fn: torch.nn.Module = self.strategy.loss_function()

        # initialize the metric registry, MUST OCCUR AFTER STRATEGY INIT
        self.metrics: MetricRegistry = self._init_metrics()

        # set up the model
        self.model: torch.nn.Module = self._init_model()

        # place logs alongside run files
        self.log_dir: Path = self.outdir / "logs"

        logger.info("%s output directory: %s", self.__class__.__name__, str(self.outdir))

        # set global seed for reproducibility
        self._init_seed()

        # register callbacks if required
        if default_callbacks:
            self._init_default_callbacks()

        # initialize the normalizer
        self.normalizer: Normalizer = self._init_normalizer()

        # initialize the optimizer
        self.optimizer: Optimizer = self._init_optimizer()

    ### INIT METHODS

    @staticmethod
    def _init_ddp(process_state: DDPProcessState | None) -> DDPProcessState | None:
        """Initialize the DDP process state handler if there is one."""
        if process_state is not None:
            process_state.attach()

        return process_state

    def _init_data(self, source: Source | SourceType) -> DataService:
        """Initialize the dataset manager."""
        # grab dataset config
        config = self.config.data

        # build the instance with config params
        manager = DataService(params=Params(config.toDict(), DataService.__name__))

        # build context and attach
        ctx = DatasetContext(source=source, process_state=self.process_state)
        manager.attach(ctx)

        return manager

    def _init_normalizer(self) -> Normalizer:
        """Initialize the normalizer."""
        # grab normalizer config
        config = self.config.normalizer

        # build the instance with config params
        params = Params(config.kwargs.toDict(), Normalizer.__name__)
        normalizer = NormalizerFactory.create(config.name, params=params)

        # build context
        ctx = NormalizerContext(data=self.data.view(StatSurface))

        # attach the normalizer
        normalizer.attach(ctx)

        # normalizer is a torch module, so need to move to device
        normalizer.to(self.device)

        return normalizer

    def _init_strategy(self) -> Strategy:
        """Initialize the strategy."""
        # grab strategy config
        config = self.config.strategy

        # build the instance with config params
        params = Params(config.kwargs.toDict(), Strategy.__name__)
        strategy = StrategyFactory.create(config.task, params=params)

        # build context
        ctx = StrategyContext(data=self.data.view(AttributeSurface))

        # attach the strategy
        strategy.attach(ctx)

        return strategy

    def _init_callback_registry(self) -> CallbackRegistry:
        return CallbackRegistry(self)

    def _init_device(self) -> torch.device:
        """Select and initialize the device."""
        # get the device selection
        if self.process_state:
            logger.debug(
                "initializing on rank %d/%d",
                self.process_state.procinfo["local_rank"], self.process_state.procinfo["world"]
            )
            device = torch.device(f"cuda:{self.process_state.procinfo['local_rank']}")
        else:
            if torch.cuda.is_available():
                logger.debug("initializing on GPU")
                device = torch.device("cuda")
            else:
                logger.debug("no accelerators found, fallback to CPU")
                device = torch.device("cpu")

        return device

    def _init_model(self) -> torch.nn.Module:
        """Initialize the model for training."""
        model_config = self.config.model

        # determine dimensions of input and output
        in_channels = self.strategy.in_channels
        out_channels = self.strategy.out_channels

        # pull hidden layers and channels from config (not controlled by strategy)
        kwargs = model_config.kwargs

        hidden_channels     = kwargs.hidden_channels
        hidden_layers       = kwargs.hidden_layers
        num_neighbors       = kwargs.num_neighbors

        logger.debug(
            "initializing %s model with in_channels=%d, out_channels=%d, kwargs=%s",
            model_config.task, in_channels, out_channels, str(kwargs.toDict())
        )

        # get the model and move to accelerator
        active_model = ModelFactory.create(
            model_config.task, in_channels, hidden_channels, out_channels, hidden_layers, num_neighbors
        )
        active_model.init_model()
        model = active_model.to(self.device)

        # if ddp is enabled, the model needs to be wrapped by torch DDP
        if self.process_state:
            model = DistributedDataParallel(
                model,
                device_ids=[self.device.index] if self.device.type == "cuda" else None,
                output_device=(self.device.index if self.device.type == "cuda" else None),
                broadcast_buffers=False,
                gradient_as_bucket_view=True,
                find_unused_parameters=False
            )

        return model

    def _init_seed(self) -> None:
        """Set a deterministic seed."""
        base_seed = self.config.seed
        rank_off = (self.process_state.procinfo["rank"] if self.process_state else 0)
        self._set_seed(base_seed + rank_off)

    def _init_metrics(self) -> MetricRegistry:
        """Initialize the trainer metric registry."""
        metric_config = self.config.metrics

        # build metrics registry
        metrics = MetricFactory.create(metric_config, self.strategy)

        return metrics

    def _init_default_callbacks(self) -> None:
        """Set up minimal default callbacks."""
        cb_spec = self.callbacks.CallbackSpec

        default_callback_specs = [
            cb_spec(callback=ExportCallback),
            cb_spec(callback=TensorBoardCallback),
            cb_spec(callback=ConsoleCallback)
        ]

        for callback in default_callback_specs:
            self.callbacks.register_spec(callback)

    def _init_optimizer(self) -> Optimizer:
        """Initialize the optimizer."""
        optim_config = self.config.optimizer
        if not hasattr(optim, optim_config.task):
            raise ValueError(f"Optimizer {optim_config.task} not found in torch.optim")

        optimizer_spec: Type[Optimizer] = getattr(optim, optim_config.task)
        optimizer = optimizer_spec(
            self.model.parameters(),
            **optim_config.kwargs.toDict()
        )

        return optimizer

    ### ---

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.exit(exc_type is KeyboardInterrupt)
        return False

    def _fire(self, hook_name: str, ctx: context.Context) -> None:
        """
        Invokes internal callback hooks but first gates by rank.

        Args:
            hook_name (str): The name of the callback method to call.
            ctx: Context to forward into the callback.
        """
        self.callbacks.fire(hook_name, ctx)

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

    def _set_split(self, split: Split) -> None:
        """Set the current split."""
        if split == Split.TRAIN:
            self.model.train()
        else:
            self.model.eval()

    def _run_split(self, dataloader: DataLoader) -> float:
        """
        Run one epoch over `dataloader`.

        Args:
            dataloader (DataLoader): Yields batches for validation or testing.
        """
        logger.info("starting split: %s, epoch=%d", self.split.value, self.current_epoch)
        start_time = time.perf_counter()

        # reset metrics for new epoch
        self.metrics.reset()

        # loss accumulator
        loss_accumulator = torch.tensor(0, dtype=torch.float64, device=self.device)
        batch_count = len(dataloader)

        # we want inference mode on eval epochs
        ctx = torch.no_grad if self.split != Split.TRAIN else nullcontext

        with ctx():
            # iterate over each batch
            for idx, batch in enumerate(dataloader):
                ctx = context.BatchBeginContext(self, batch=batch)
                self._fire("on_batch_begin", ctx)

                # move the batch to the accelerator
                batch = batch.to(self.device, non_blocking=True)

                # dispatch to normalizer after move but before callback hook fire
                batch.x = self.normalizer(batch.x, ModelInputRole.FEATURES)
                if self.strategy.name == "regression":
                    batch.y = self.normalizer(batch.y, ModelInputRole.LABELS)

                ctx = context.BatchTransferContext(self, batch=batch)
                self._fire("on_batch_transfer", ctx)

                # forward pass through model
                out = self.model(batch.x, batch.batch)  # type: ignore[arg-type]
                target = self.strategy.adapt_targets(batch, out)

                # compute batch loss and accumulate without sync
                loss = self.loss_fn(out, target)
                loss_accumulator += loss.detach()

                # pass results to metrics for update
                self.metrics.update(out.detach(), target.detach())

                # dont want to update weights or run backward pass on eval
                if self.split == Split.TRAIN:
                    # dont want to accumulate gradients
                    self.optimizer.zero_grad()

                    # run backward pass
                    loss.backward()

                    # step the optimizer to update weights
                    self.optimizer.step()

                # need to inverse normalize out and target before calling on_batch_end callback hook
                # this is so plotters and other callbacks can see raw unnormalized values
                # only needs to be done on eval loops
                if self.split in Split.eval():
                    out     = self.normalizer(out,      ModelInputRole.LABELS, inverse=True)
                    target  = self.normalizer(target,   ModelInputRole.LABELS, inverse=True)

                ctx = context.BatchEndContext(
                    self, batch=batch, out=out.detach(), target=target.detach(), loss=loss.detach()
                )
                self._fire("on_batch_end", ctx)

            # update metric summaries on eval steps
            if self.split in Split.eval():
                self.metrics.update_summaries()

        # accumulated epoch loss
        epoch_loss = loss_accumulator.item() / batch_count if batch_count > 0 else float("nan")

        # time the execution
        elapsed = time.perf_counter() - start_time

        # log complete split with loss and elapsed time
        logging.info("completed split: loss=%.5g, elapsed=%.5g", epoch_loss, elapsed)

        return epoch_loss

    def _train(self) -> None:
        """Run a single training epoch."""
        ctx = context.TrainBeginContext(self)
        self._fire("on_train_begin", ctx)

        # set the trainer mode
        self._set_split(Split.TRAIN)

        # grab dataloader from registry and run epoch
        dataloader = self.data.dataloader(Split.TRAIN)
        loss = self._run_split(dataloader)

        ctx = context.TrainEndContext(self, loss)
        self._fire("on_train_end", ctx)

    def _validate(self) -> None:
        """
        Compute validation metrics without altering model weights.
        """
        ctx = context.ValidationBeginContext(self)
        self._fire("on_validation_begin", ctx)

        # set the trainer mode
        self._set_split(Split.VAL)

        # grab dataloader from data and run epoch
        dataloader = self.data.dataloader(Split.VAL)
        loss = self._run_split(dataloader)

        ctx = context.ValidationEndContext(self, loss)
        self._fire("on_validation_end", ctx)

    def _test(self) -> None:
        """
        Compute test metrics using the final model (no weight updates).
        """
        ctx = context.TestBeginContext(self)
        self._fire("on_test_begin", ctx)

        # set the trainer mode
        self._set_split(Split.TEST)

        # grab dataloader from data and run epoch
        dataloader = self.data.dataloader(Split.TEST)
        loss = self._run_split(dataloader)

        ctx = context.TestEndContext(self, loss)
        self._fire("on_test_end", ctx)

    def execute(self) -> None:
        """
        Method for executing the full pipeline: training, validation at set intervals, final testing, and teardown.
        """
        logger.info("executing training loop")

        # fire on_execute callback hook
        ctx = context.ExecuteContext(self)
        self._fire("on_execute", ctx)

        # iterate over epochs
        # self.current_epoch is always set before any usage, so not checking via hasattr is fine here
        for self.current_epoch in range(self.config.trainer.max_epochs):
            ctx = context.EpochBeginContext(self)
            self._fire("on_epoch_begin", ctx)

            # set sampler epoch
            self.data.set_epoch(self.current_epoch)

            # train on all ranks
            self._train()

            # validate only on main rank
            if (
                (not self.process_state or self.process_state.is_main_process())
                and self.config.trainer.val_interval > 0
                and (self.current_epoch + 1) % self.config.trainer.val_interval == 0
            ):
                self._validate()

            # test only once at end, on main rank
            if (
                (not self.process_state or self.process_state.is_main_process())
                and (self.current_epoch + 1) == self.config.trainer.max_epochs
            ):
                self._test()

            ctx = context.EpochEndContext(self)
            self._fire("on_epoch_end", ctx)

            if self.process_state:
                self.process_state.barrier()  # keep ranks in lockstep (no-op if not DDP)

    def exit(self, interrupt: bool) -> None:
        try:
            ctx = context.TeardownContext(self)
            self._fire("on_teardown", ctx)
        finally:
            if self.process_state:
                self.process_state.cleanup(interrupt)
