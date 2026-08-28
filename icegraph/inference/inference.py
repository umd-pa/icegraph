# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Any
from typing_extensions import override
from pathlib import Path
from functools import cached_property
import time

import torch
from torch import Tensor
import numpy as np

from icegraph.common.data import DataRole, GraphBatch
from icegraph.common.engine import ComponentKind

from icegraph.engine import Engine

# callbacks
from .callbacks import context

# config
from .config import InferenceConfig

# module logger
import logging
logger = logging.getLogger(__name__)

__all__ = ["BatchInference"]


class BatchInference(Engine[InferenceConfig]):

    def __init__(self, config: InferenceConfig, *, state_dicts: dict[str, dict[str, Any]]) -> None:
        """Initialize the Trainer."""
        logger.debug("initializing %s", type(self).__name__)
        super().__init__(config)

        # stash outdir
        self.outdir = self.config.outdir

        self._load_state_dicts(state_dicts)

    @classmethod
    @override
    def from_config(cls, config: dict[str, Any]) -> BatchInference:
        # manually extract model configs to inject into provided config
        model_path = config.get("model_path")

        if not isinstance(model_path, str):
            raise ValueError(
                f"Invalid 'model_path' from config, expected a path (str), "
                + f"got {model_path} ({type(model_path).__name__})"
            )

        if not Path(model_path).exists():
            raise FileNotFoundError(f"Invalid 'model_path' from config, file {model_path} does not exist.")

        # load the model
        state: dict[str, Any] = torch.load(
            model_path,
            map_location="cpu",
            weights_only=False
        )

        # extract component state dicts and configs
        config.update(state["config"])

        return cls(InferenceConfig(**config), state_dicts=state["states"])

    # services

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
    def data(self):
        return self.services.require("data", required_by=type(self))

    # components

    @cached_property
    def model(self):
        return self.components.require(ComponentKind.MODEL, required_by=type(self))

    @cached_property
    def normalizer(self):
        return self.components.require(ComponentKind.NORMALIZER, required_by=type(self))

    @cached_property
    def transformer(self):
        return self.components.require(ComponentKind.TRANSFORMER, required_by=type(self))

    @cached_property
    def edges(self):
        return self.components.require(ComponentKind.EDGES, required_by=type(self))

    @torch.no_grad()
    def _process_batch(self, batch: GraphBatch) -> Tensor:
        # pull from batch
        features = batch.features
        index       = batch.batch

        # build connectivity while the feature block still holds raw values
        edge_index, edge_attr = self.edges(features, batch=index)

        features = self.transformer(features, DataRole.FEATURES)
        features = self.normalizer(features, DataRole.FEATURES)

        out = self.model(features, edge_index=edge_index, edge_attr=edge_attr, batch=index)

        out = self.normalizer(out, DataRole.TARGETS, inverse=True)
        out = self.transformer(out, DataRole.TARGETS, inverse=True)

        return out

    def _predict(self) -> None:
        logger.info("starting predictions")

        # get dataloader
        dataloader = self.get_dataloader()

        # iterate over each batch
        for b in dataloader:
            # move to device
            raw_batch = b.to_device(self.state.device, non_blocking=True)

            # convert to graph batch
            graph_batch = GraphBatch.from_raw_batch(raw_batch, self.decode.get_segment_layout)

            # process batch
            start_time = time.perf_counter()

            _ = self._process_batch(graph_batch)

            # time the execution
            elapsed = time.perf_counter() - start_time

            # log elapsed time
            logger.debug("completed predictions for batch in %f s",  elapsed)

        return None

    def get_dataloader(self):  # allow to infer type
        # for inference, no splits and all keys loaded in one loader
        keys = np.arange(len(self.record), dtype=np.int64)

        # no need for targets in inference
        # technically simweights, auxiliary not needed as well, but allow them if user wants
        # to include for plotting, inference with edges, etc
        exclude_roles = [DataRole.TARGETS]

        # build spec and return dataloader, no need to cache spec since only iterating over loader once
        spec = self.data.loader_spec.make(keys, exclude_roles=exclude_roles)
        return self.data.dataloader(spec)

    @override
    def execute(self) -> None:
        """Execute inference."""
        logger.info("executing inference")

        # fire on_execute callback hook
        ctx = context.ExecuteContext(self)
        self.callbacks.fire("on_execute", ctx)

        # run prediction
        self._predict()

    @override
    def close(self) -> None:
        logger.info("shutting down inference engine")
        try:
            ctx = context.TeardownContext(self)
            self.callbacks.fire("on_teardown", ctx)
        finally:
            self.services.close()
