# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Any
from typing_extensions import override
from pathlib import Path
from contextlib import nullcontext
import time
from functools import cached_property, lru_cache

# torch imports
import torch
from torch import Tensor
from torch_geometric.seed import seed_everything

# data container
from icegraph.common.data import GraphBatch, ProcessedGraphBatch

from icegraph.common.data import Split, DataRole
from icegraph.common.tensors import SegmentedTensor
from icegraph.common.engine import ComponentKind

from ..engine import Engine

# callbacks
from .callbacks import context

# config
from .config import TrainerConfig

__all__ = ["Trainer"]


# module logger
import logging
logger = logging.getLogger(__name__)


class Trainer(Engine[TrainerConfig]):
    """
    Engine responsible for managing the training, validation, and testing
    lifecycle of a PyTorch model.
    """

    def __init__(self, config: TrainerConfig) -> None:
        """Initialize the Trainer."""
        logger.debug("initializing %s", type(self).__name__)
        super().__init__(config)

        # stash outdir and derive logdir
        self.outdir: Path = Path(self.config.outdir)
        self.logdir: Path = self.outdir / "logs"
        self.plotdir: Path = self.outdir / "plots"

        # global access to current epoch
        self.current_epoch: int = 0

        # global access to current split
        self.split: Split = Split.TRAIN  # first split is always train

        # set global seed for reproducibility
        self._set_seed(self.state.seed + self.state.rank)

    @classmethod
    @override
    def from_config(cls, config: dict[str, Any]) -> Trainer:
        return cls(config=TrainerConfig(**config))

    # cached properties for potentially very slightly faster hot path access, but mostly convenience
    # going to allow each to derive return type to avoid a bunch of garbo imports

    # built in services

    @cached_property
    def state(self):
        return self.services.require("state", required_by=type(self))

    @cached_property
    def decode(self):
        return self.services.require("decode", required_by=type(self))

    @cached_property
    def record(self):
        return self.services.require("record", required_by=type(self))

    @cached_property
    def metrics(self):
        return self.services.require("metrics", required_by=type(self))

    @cached_property
    def data(self):
        return self.services.require("data", required_by=type(self))

    # components

    @cached_property
    def model(self):
        return self.components.require(ComponentKind.MODEL, required_by=type(self))

    @cached_property
    def optimizer(self):
        return self.components.require(ComponentKind.OPTIMIZER, required_by=type(self))

    @cached_property
    def loss(self):
        return self.components.require(ComponentKind.LOSS, required_by=type(self))

    @cached_property
    def normalizer(self):
        return self.components.require(ComponentKind.NORMALIZER, required_by=type(self))

    @cached_property
    def transformer(self):
        return self.components.require(ComponentKind.TRANSFORMER, required_by=type(self))

    @cached_property
    def edges(self):
        return self.components.require(ComponentKind.EDGES, required_by=type(self))

    # training algo

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

    def _process_batch(self, batch: GraphBatch, split: Split) -> Tensor:
        # call on batch begin hook
        ctx = context.BatchBeginContext(engine=self, batch=batch)
        self.callbacks.fire("on_batch_begin", ctx)

        # pull from batch
        features    = batch.features
        targets     = batch.targets
        index       = batch.batch

        # build connectivity while the feature block still holds raw values:
        # the transformer and normalizer rescale columns independently, which
        # would distort the space the neighbour search runs in
        edge_index, edge_attr = self.edges(features, batch=index)

        # transform and normalize features on accelerator
        features = self.transformer(features, DataRole.FEATURES)
        features = self.normalizer(features, DataRole.FEATURES)

        # normalize targets on accelerator if required by policy
        targets = self.transformer(targets, DataRole.TARGETS)
        targets = self.normalizer(targets, DataRole.TARGETS)

        # forward pass
        out: SegmentedTensor = self.model(features, edge_index=edge_index, edge_attr=edge_attr, batch=index)

        # compute batch loss
        loss = self.loss(out, targets)

        # run backward pass if in training
        if split == Split.TRAIN:
            # dont want to accumulate gradients
            self.optimizer.zero_grad()

            # run backward pass
            loss.backward()

            # step the optimizer to update weights
            self.optimizer.step()

        # update metrics
        self.metrics.update(out.detach(), targets.detach(), split)  # use out and targets which are normalized

        # denorm out if required by policy
        out = self.normalizer(out, DataRole.TARGETS, inverse=True)
        out = self.transformer(out, DataRole.TARGETS, inverse=True)

        # attach out to the batch, then detach each tensor from autograd
        processed_batch = ProcessedGraphBatch.from_graph_batch(batch, out=out)
        processed_batch = processed_batch.detach()

        # call on batch end hook
        ctx = context.BatchEndContext(engine=self, batch=processed_batch, loss=loss.detach())
        self.callbacks.fire("on_batch_end", ctx)

        return loss.detach()

    def _run_split(self, split: Split) -> Tensor:
        """Run one epoch for a given split"""
        logger.info("starting split: %s, epoch=%d", split.value, self.current_epoch + 1)
        start_time = time.perf_counter()

        # declare global split
        self.split = split

        # retrieve the dataloader for current split
        dataloader = self.get_dataloader(split)

        # reset metrics for new epoch
        self.metrics.reset(split)

        # loss accumulator
        loss_accumulator = torch.tensor(0, dtype=torch.float32, device=self.state.device)
        batch_count = len(dataloader)

        # dtype map for batch casting targets
        dtype_map = {
            DataRole.TARGETS: self.policy.task_spec.target_dtype
        } if self.policy is not None else None

        # we want no grad on eval epochs
        ctx: torch.no_grad | nullcontext[None] = torch.no_grad() if split in Split.eval() else nullcontext()

        # ensure model is in correct mode
        if split in Split.eval():
            _ = self.model.eval()
        else:
            _ = self.model.train()

        with ctx:
            # iterate over each batch
            for b in dataloader:
                # move to device
                raw_batch = b.to_device(self.state.device, non_blocking=True)

                # convert to graph batch
                graph_batch = GraphBatch.from_raw_batch(raw_batch, self.decode.get_segment_layout)

                # trainer responsible for batch dtype cast
                if dtype_map:
                    graph_batch = graph_batch.to_dtype(dtype_map)

                # process batch and accumulate loss
                loss_accumulator += self._process_batch(graph_batch, split)

            # update metric summaries
            self.metrics.update_summaries(split)

        # accumulated epoch loss
        # one sync per epoch, not too bad
        epoch_loss = (loss_accumulator / batch_count if batch_count > 0 else torch.tensor(float("nan"))).cpu()

        # time the execution
        elapsed = time.perf_counter() - start_time

        # log complete split with loss and elapsed time
        logging.info("completed split: loss=%.5g, elapsed=%.5g", float(epoch_loss.item()), elapsed)

        return epoch_loss

    @lru_cache(maxsize=None)
    def _get_loader_spec(self, split: Split):  # allow to infer type
        # this needs to me memoized, each spec is both the initial build spec
        # and the key to access the built dataloader later
        keys = self.decode.get_keys(split)
        exclude_roles = [DataRole.AUXILIARY] if split not in Split.eval() else None

        return self.data.loader_spec.make(keys, exclude_roles=exclude_roles)

    def get_dataloader(self, split: Split):  # allow to infer type
        return self.data.dataloader(self._get_loader_spec(split))

    def _run_training_epoch(self) -> None:
        """Run a single training epoch."""
        ctx = context.TrainBeginContext(engine=self)
        self.callbacks.fire("on_train_begin", ctx)

        # run split
        loss = self._run_split(Split.TRAIN)

        ctx = context.TrainEndContext(engine=self, loss=loss)
        self.callbacks.fire("on_train_end", ctx)

    def _run_validation_epoch(self) -> None:
        """Run a single validation epoch."""
        ctx = context.ValidationBeginContext(engine=self)
        self.callbacks.fire("on_validation_begin", ctx)

        # run split
        loss = self._run_split(Split.VAL)

        ctx = context.ValidationEndContext(engine=self, loss=loss)
        self.callbacks.fire("on_validation_end", ctx)

    def _run_test_epoch(self) -> None:
        """Run a single test epoch."""
        ctx = context.TestBeginContext(engine=self)
        self.callbacks.fire("on_test_begin", ctx)

        # run split
        loss = self._run_split(Split.TEST)

        ctx = context.TestEndContext(engine=self, loss=loss)
        self.callbacks.fire("on_test_end", ctx)

    @override
    def execute(self) -> None:
        """
        Method for executing the full pipeline: training, validation at set intervals, final testing, and teardown.
        """
        logger.info("executing training loop")

        # fire on_execute callback hook
        ctx = context.ExecuteContext(engine=self)
        self.callbacks.fire("on_execute", ctx)

        # iterate over epochs
        # self.current_epoch is always set before any usage, so not checking via hasattr is fine here
        for self.current_epoch in range(self.config.max_epochs):
            ctx = context.EpochBeginContext(engine=self)
            self.callbacks.fire("on_epoch_begin", ctx)

            # set sampler epoch
            self.data.set_epoch(self.current_epoch)

            # train on all ranks
            self._run_training_epoch()

            # validate only on main rank
            if (
                self.state.is_main_process()
                and self.config.val_interval > 0
                and (self.current_epoch + 1) % self.config.val_interval == 0
            ):
                self._run_validation_epoch()

            # test only once at end, on main rank
            if (
                self.state.is_main_process()
                and (self.current_epoch + 1) == self.config.max_epochs
            ):
                self._run_test_epoch()

            ctx = context.EpochEndContext(engine=self)
            self.callbacks.fire("on_epoch_end", ctx)

            self.state.barrier()  # keep ranks in lockstep (no-op if not DDP)

    @override
    def close(self) -> None:
        try:
            ctx = context.TeardownContext(engine=self)
            self.callbacks.fire("on_teardown", ctx)
        finally:
            super().close()
