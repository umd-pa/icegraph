# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Self, Type, Any, TypeVar
from pathlib import Path
from contextlib import nullcontext
import time

# torch imports
import torch
from torch_geometric.seed import seed_everything
from torch_geometric.loader import DataLoader
from torch.nn.parallel import DistributedDataParallel  # DDP

# types
from icegraph.types.data import Split, ModelInputRole
from icegraph.types.factory import Factory
from icegraph.types.files import SourceType, Source

from .config import ConfigStore

# services
from .services import ServiceManager

# callbacks
from .callbacks import CallbackManager, context

# components
from .components import ComponentContext, Component

from .components.normalizer import Normalizer, NormalizerFactory, NormalizerContext
from .components.model import Model, ModelFactory, ModelContext
from .components.loss import LossFunction, LossFactory, LossContext
from .components.optimizer import Optimizer, OptimizerFactory, OptimizerContext

__all__ = ["Trainer"]


# module logger
import logging
logger = logging.getLogger(__name__)


T = TypeVar("T", bound=Component)

class Trainer:
    """
    Trainer class responsible for managing the training, validation, and testing
    lifecycle of a PyTorch model.

    This class is the central coordinator for all training processes, including
    model execution, optimization, metrics, etc.
    """

    def __init__(
        self,
        source: Source | SourceType,
        outdir: str | Path
    ) -> None:
        """
        Initialize the Trainer.

        Args:
            source (SourceType | Source): Source files to train from.
            outdir (str | Path): Path to save the trained model and any other generated files.
        """
        # call to super
        super().__init__()

        logger.debug("initializing %s", self.__class__.__name__)

        self.source = source

        # stash outdir and derive logdir
        self.outdir = Path(outdir)
        self.logdir = self.outdir / "logs"

        # load trainer config
        self.config = ConfigStore.get()

        # global access to current epoch
        self.current_epoch: int = 0

        # slots for active mode and split
        self.split: Split = Split.TRAIN  # first split is always train

        # initialize the service manager
        self.services = ServiceManager.from_config(self, self.config.services.toDict())

        self.state      = self.services.require("state",    required_by=Trainer)
        self.metrics    = self.services.require("metrics",  required_by=Trainer)
        self.strategy   = self.services.require("strategy", required_by=Trainer)
        self.data       = self.services.require("data",     required_by=Trainer)

        # set global seed for reproducibility
        self._set_seed(self.config.seed + self.state.rank)

        # initialize components
        self.loss:          LossFunction    = self._init_loss()
        self.model:         Model           = self._init_model()
        self.normalizer:    Normalizer      = self._init_normalizer()
        self.optimizer:     Optimizer       = self._init_optimizer()

        # initialize the callback manager
        self.callbacks = CallbackManager(self)

    ### INIT METHODS

    def _construct_component(self, factory: type[Factory[T]], config: dict[str, Any], ctx: ComponentContext) -> T:
        component = factory.create(config["name"], config=config["kwargs"])

        # attach the component using context
        component.attach(ctx)

        # move to device
        component.to(self.state.device)

        return component

    def _init_loss(self) -> LossFunction[Any]:
        """Initialize the normalizer."""
        # grab loss config
        config = self.config.components.loss.model_dump()
        ctx = LossContext(services=self.services)
        loss = self._construct_component(LossFactory, config, ctx)

        # ensure selected loss is compatible with declared strategy
        self.strategy.ensure_compatible(loss)

        return loss

    def _init_normalizer(self) -> Normalizer[Any]:
        """Initialize the normalizer."""
        # grab normalizer config
        config = self.config.components.normalizer.model_dump()
        ctx = NormalizerContext(services=self.services)
        normalizer = self._construct_component(NormalizerFactory, config, ctx)

        # ensure selected normalizer is compatible with declared strategy
        self.strategy.ensure_compatible(normalizer)

        return normalizer

    def _init_model(self) -> Model[Any]:
        """Initialize the model for training."""
        config = self.config.components.model.model_dump()
        ctx = ModelContext(services=self.services)
        model = self._construct_component(ModelFactory, config, ctx)

        # ensure selected model is compatible with declared strategy
        self.strategy.ensure_compatible(model)

        # if ddp is enabled, the model needs to be wrapped by torch DDP
        # get state service
        state = self.services.require("state", required_by=Trainer)
        if state.is_ddp():
            model = DistributedDataParallel(
                model,
                device_ids=[state.device.index] if state.device.type == "cuda" else None,
                output_device=(state.device.index if state.device.type == "cuda" else None),
                broadcast_buffers=False,
                gradient_as_bucket_view=True,
                find_unused_parameters=False
            )

        return model

    def _init_optimizer(self) -> Optimizer[Any]:
        """Initialize the normalizer."""
        # grab normalizer config
        config = self.config.components.optimizer.model_dump()
        ctx = OptimizerContext(services=self.services, model_params=self.model.parameters())
        optimizer = self._construct_component(OptimizerFactory, config, ctx)

        # ensure selected optimizer is compatible with declared strategy
        self.strategy.ensure_compatible(optimizer)

        return optimizer

    ### ---

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
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

        # get metrics service
        metrics = self.services.require("metrics", required_by=Trainer)

        # reset metrics for new epoch
        metrics.reset()

        # loss accumulator
        loss_accumulator = torch.tensor(0, dtype=torch.float64, device=self.state.device)
        batch_count = len(dataloader)

        # we want inference mode on eval epochs
        ctx = torch.no_grad if self.split in Split.eval() else nullcontext

        with ctx():
            # iterate over each batch
            for idx, batch in enumerate(dataloader):
                ctx = context.BatchBeginContext(self, batch=batch)
                self._fire("on_batch_begin", ctx)

                # move the batch to the accelerator
                batch = batch.to(self.state.device, non_blocking=True)

                # dispatch to normalizer after move but before callback hook fire
                batch.x = self.normalizer(batch.x, ModelInputRole.FEATURES)
                if self.strategy.mode == "regression":
                    batch.y = self.normalizer(batch.y, ModelInputRole.LABELS)

                ctx = context.BatchTransferContext(self, batch=batch)
                self._fire("on_batch_transfer", ctx)

                # forward pass through model
                out = self.model(batch.x, batch.batch)  # type: ignore[arg-type]
                target = self.strategy.adapt_targets(batch.y)

                # compute batch loss and accumulate without sync
                loss = self.loss(out, target)
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
                metrics.update_summaries()

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

        # grab dataloader from data service
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
                self.state.is_main_process()
                and self.config.trainer.val_interval > 0
                and (self.current_epoch + 1) % self.config.trainer.val_interval == 0
            ):
                self._validate()

            # test only once at end, on main rank
            if (
                self.state.is_main_process()
                and (self.current_epoch + 1) == self.config.trainer.max_epochs
            ):
                self._test()

            ctx = context.EpochEndContext(self)
            self._fire("on_epoch_end", ctx)

            self.state.barrier()  # keep ranks in lockstep (no-op if not DDP)

    def close(self) -> None:
        try:
            ctx = context.TeardownContext(self)
            self._fire("on_teardown", ctx)
        finally:
            self.state.close()
