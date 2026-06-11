# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Self, Any, TypeVar, TYPE_CHECKING
from pathlib import Path
from contextlib import nullcontext
import time

import yaml

# torch imports
import torch
from torch import Tensor
from torch_geometric.seed import seed_everything
from torch_geometric.loader import DataLoader

# data container
from icegraph.common.data import GraphBatch, ProcessedGraphBatch, RawGraphBatch

# types
from icegraph.common.data import Split, DataRole
from icegraph.common.factory import PluginFactory
from icegraph.common.engine import Engine
from icegraph.common.tensors import SegmentedTensor

# services
from icegraph.engine.services import ServiceManager

# components
from icegraph.engine.components import ComponentContext, Component

from icegraph.engine.components.adapter import Adapter, AdapterFactory, AdapterContext
from icegraph.engine.components.normalizer import Normalizer, NormalizerFactory, NormalizerContext
from icegraph.engine.components.model import Model, ModelFactory, ModelContext
from icegraph.engine.components.loss import LossFunction, LossFactory, LossContext
from icegraph.engine.components.optimizer import Optimizer, OptimizerFactory, OptimizerContext
from icegraph.engine.components.transformer import Transformer, TransformerFactory, TransformerContext

# callbacks
from .callbacks import CallbackManager, context

# config
from .config import TrainerConfig, ComponentOption

if TYPE_CHECKING:
    from icegraph.common.plugins import Plugin

__all__ = ["Trainer"]


# module logger
import logging
logger = logging.getLogger(__name__)


T = TypeVar("T", bound=Component[Any, Any])


class Trainer(Engine):
    """
    Trainer class responsible for managing the training, validation, and testing
    lifecycle of a PyTorch model.

    This class is the central coordinator for all training processes, including
    model execution, optimization, metrics, etc.
    """

    def __init__(self, *, config: dict[str, Any]) -> None:
        """Initialize the Trainer."""
        logger.debug("initializing %s", type(self).__name__)

        # load trainer config
        self.config = TrainerConfig(**config)

        # stash outdir and derive logdir
        self.outdir = Path(self.config.outdir)
        self.logdir = self.outdir / "logs"

        # global access to current epoch
        self.current_epoch: int = 0

        # slots for active split
        self.split: Split = Split.TRAIN  # first split is always train

        # extract policy
        self.policy = self.config.policy

        # initialize the service manager
        self.services   = ServiceManager.from_config(self, dict(self.config.services))

        # get defaults
        self.state      = self.services.require("state",    required_by=Trainer)
        self.metrics    = self.services.require("metrics",  required_by=Trainer)
        self.data       = self.services.require("data",     required_by=Trainer)
        self.record     = self.services.require("record",   required_by=Trainer)
        self.decode     = self.services.require("decode",   required_by=Trainer)

        # set global seed for reproducibility
        self._set_seed(self.config.seed + self.state.rank)

        # build components ordered properly
        self.adapter        = self._init_adapter()
        self.transformer    = self._init_transformer()

        self.loss           = self._init_loss()
        self.model          = self._init_model()
        self.normalizer     = self._init_normalizer()

        self.optimizer      = self._init_optimizer()

        # initialize the callback manager
        self.callbacks = CallbackManager(self)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        with Path(path).open("r") as f:
            return cls(config=yaml.safe_load(f))

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> Self:
        pass

    def checkpoint(self, path: str | Path) -> None:
        pass

    ### POLICY HANDLER

    def ensure_compatible(self, p: Plugin, /) -> None:
        if not p.compatible:
            # an empty compatibility tuple indicates compatibility with any policy
            return

        if self.policy not in p.compatible:
            raise RuntimeError(f"Module '{type(p).__name__}' is not compatible with policy '{self.policy}'.")

    ### INIT METHODS

    def _construct_component(
            self, factory: type[PluginFactory[T]], config: ComponentOption, ctx: ComponentContext
    ) -> T:
        component = factory.create(config.name, **config.kwargs)

        # attach the component using context
        component.attach(ctx)

        # move to device
        component.to(self.state.device)

        # ensure component is compatible with selected policy
        self.ensure_compatible(component)

        return component

    def _init_adapter(self) -> Adapter[Any]:
        """Initialize the adapter for set policy."""
        # build directly since using policy not configurations
        config = self.config.components.adapter
        adapter = AdapterFactory.create(self.policy, **config.kwargs)

        # attach the adapter
        ctx = AdapterContext(self.services, debug=self.config.debug)
        adapter.attach(ctx)

        # adapter stays on cpu, so no move
        return adapter

    def _init_transformer(self) -> Transformer[Any]:
        """Initialize the transformer."""
        # grab transformer config
        config = self.config.components.transformer
        ctx = TransformerContext(
            services=self.services,
            contract=self.adapter.transformer_contract(),
            debug=self.config.debug
        )

        return self._construct_component(TransformerFactory, config, ctx)

    def _init_loss(self) -> LossFunction[Any]:
        """Initialize the loss function."""
        # grab loss config
        config = self.config.components.loss
        ctx = LossContext(
            services=self.services,
            contract=self.adapter.loss_contract(),
            debug=self.config.debug
        )

        return self._construct_component(LossFactory, config, ctx)

    def _init_normalizer(self) -> Normalizer[Any]:
        """Initialize the normalizer."""
        # grab normalizer config
        config = self.config.components.normalizer
        ctx = NormalizerContext(
            services=self.services,
            transformer_spec_list=self.transformer.spec_list,
            contract=self.adapter.normalizer_contract(),
            debug=self.config.debug
        )

        return self._construct_component(NormalizerFactory, config, ctx)

    def _init_optimizer(self) -> Optimizer[Any]:
        """Initialize the optimizer."""
        # grab optimizer config
        config = self.config.components.optimizer
        # optimizer is not a contract component, no contract
        ctx = OptimizerContext(
            services=self.services,
            model_params=self.model.parameters(),
            debug=self.config.debug
        )

        return self._construct_component(OptimizerFactory, config, ctx)

    def _init_model(self) -> Model[Any]:
        """Initialize the model for training."""
        # grab model config
        config = self.config.components.model
        ctx = ModelContext(
            services=self.services,
            contract=self.adapter.model_contract(),
            debug=self.config.debug
        )
        model = self._construct_component(ModelFactory, config, ctx)

        # model needs to be bound to execution context
        self.state.bind_model(model)

        return model

    ### ---

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False

    def _fire(self, hook_name: str, ctx: context.Context) -> None:
        """
        Invokes internal callback hooks.

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
        self.split = split

        if split == Split.TRAIN:
            self.model.train()
        else:
            self.model.eval()

    def _process_batch(self, batch: GraphBatch) -> Tensor:
        # call on batch begin hook
        ctx = context.BatchBeginContext(self, batch=batch)
        self._fire("on_batch_begin", ctx)

        # adapt batch using adapter hook
        batch = self.adapter.preprocess_batch(batch)

        # pull from batch
        features    = batch.features
        targets     = batch.targets
        index       = batch.batch

        # transform and normalize features on accelerator
        features = self.transformer(features, DataRole.FEATURES)
        features = self.normalizer(features, DataRole.FEATURES)

        # normalize targets on accelerator if required by policy
        if self.adapter.use_normalized_targets:
            targets = self.transformer(targets, DataRole.TARGETS)
            targets = self.normalizer(targets, DataRole.TARGETS)

        # forward pass
        out: SegmentedTensor = self.model(features, index)

        # compute batch loss
        loss = self.loss(out, targets)

        # run backward pass if in training
        if self.split == Split.TRAIN:
            # dont want to accumulate gradients
            self.optimizer.zero_grad()

            # run backward pass
            loss.backward()

            # step the optimizer to update weights
            self.optimizer.step()

        # update metrics
        self.metrics.update(out.detach(), targets.detach())  # use out and targets which are normalized

        # denorm out if required by policy
        if self.adapter.use_normalized_targets:
            out = self.normalizer(out, DataRole.TARGETS, inverse=True)
            out = self.transformer(out, DataRole.TARGETS, inverse=True)

        # attach out to the batch, then detach each tensor from autograd
        processed_batch = ProcessedGraphBatch.from_graph_batch(batch, out=out)
        processed_batch = processed_batch.detach()

        # call on batch end hook
        ctx = context.BatchEndContext(self, batch=processed_batch, loss=loss.detach())
        self._fire("on_batch_end", ctx)

        return loss.detach()

    def _run_split(self, dataloader: DataLoader) -> float:
        """
        Run one epoch over `dataloader`.

        Args:
            dataloader (DataLoader): Yields batches for validation or testing.
        """
        logger.info("starting split: %s, epoch=%d", self.split.value, self.current_epoch + 1)
        start_time = time.perf_counter()

        # reset metrics for new epoch
        self.metrics.reset()

        # loss accumulator
        loss_accumulator = torch.tensor(0, dtype=torch.float32, device=self.state.device)
        batch_count = len(dataloader)

        # we want no grad on eval epochs
        ctx = torch.no_grad() if self.split in Split.eval() else nullcontext()

        with ctx:
            # iterate over each batch
            for raw_batch in dataloader:
                raw_batch: RawGraphBatch

                # move to device
                raw_batch = raw_batch.to_device(self.state.device, non_blocking=True)

                # convert to graph batch
                get_layout = self.decode.get_segment_layout
                graph_batch = GraphBatch.from_raw_batch(raw_batch, get_layout)

                # process batch and accumulate loss
                loss_accumulator += self._process_batch(graph_batch)

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
        for self.current_epoch in range(self.config.max_epochs):
            ctx = context.EpochBeginContext(self)
            self._fire("on_epoch_begin", ctx)

            # set sampler epoch
            self.data.set_epoch(self.current_epoch)

            # train on all ranks
            self._train()

            # validate only on main rank
            if (
                self.state.is_main_process()
                and self.config.val_interval > 0
                and (self.current_epoch + 1) % self.config.val_interval == 0
            ):
                self._validate()

            # test only once at end, on main rank
            if (
                self.state.is_main_process()
                and (self.current_epoch + 1) == self.config.max_epochs
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
            self.services.close()
