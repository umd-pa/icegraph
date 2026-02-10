# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from abc import ABC, abstractmethod
from typing import Any, Sequence, cast, Generic, TypeVar

import torch
from torch import Tensor
from torch.nn import Module, ModuleList, ModuleDict
import torch_geometric

from ..model import Model

__all__ = ["BlockModel"]


C = TypeVar("C")


# model config is handled by each individual implementation, as it can vary wildly
# thus this class is still generic in C
class BlockModel(Model[C], ABC, Generic[C]):

    # make the type checker happy
    _blocks:        ModuleList | None
    _activation:    Module | None
    _out:           Module | None
    _in_channels:   int | None
    _out_channels:  int | None

    def build(self) -> None:
        # cache for activation and blocks
        self._blocks        = None
        self._activation    = None
        self._out           = None

        # cache for in and out channels set by the strategy
        self._in_channels   = None
        self._out_channels  = None

    def on_attach(self) -> None:
        # get strategy service
        strategy = self._ctx.services.require("strategy", required_by=type(self))

        self._in_channels   = strategy.in_channels
        self._out_channels  = strategy.out_channels

    def forward(self, t: Tensor, /, batch: Tensor | None = None) -> Tensor:
        """
        Forward pass through model.

        Args:
            t (Tensor): Node feature matrix with shape [num_nodes, in_channels].
            batch (Tensor | None): Batch vector which maps each node to its graph in the batch.
        """
        if batch is None:
            # If batch is not provided (single-sample inference), create a dummy batch tensor.
            # this assumes all nodes belong to a single graph
            batch = torch.zeros(t.size(0), dtype=torch.long, device=t.device)

        # forward pass through each block
        for block in self.blocks:
            t = self._forward_block(t, batch, block, self.activation)

        t = torch_geometric.nn.global_mean_pool(t, batch)
        return self.out(t)

    @property
    def out(self) -> Module:
        if self._out is None:
            self._out = self._build_out(self.config, self._in_channels, self._out_channels)
        return self._out

    @property
    def blocks(self) -> Sequence[ModuleDict]:
        if self._blocks is None:
            self._blocks = ModuleList(self._build_blocks(self.config, self._in_channels, self._out_channels))
        return cast(Sequence[ModuleDict], cast(object, self._blocks))

    @property
    def activation(self) -> Module:
        if self._activation is None:
            self._activation = self._build_activation(self.config)
        return self._activation

    @abstractmethod
    def _build_blocks(self, config: C, in_channels: int, out_channels: int) -> list[ModuleDict]:
        ...

    @abstractmethod
    def _build_activation(self, config: C) -> Module:
        ...

    @abstractmethod
    def _build_out(self, config: C, in_channels: int, out_channels: int) -> Module:
        ...

    @abstractmethod
    def _forward_block(self, tensor: Tensor, batch: Tensor, block: ModuleDict, activation: Module) -> Tensor:
        ...

    def get_extra_state(self) -> dict[str, Any]:
        # Must be picklable
        return {"config": self.config.model_dump(mode="json")}

    def set_extra_state(self, state: dict[str, Any]) -> None:
        self.config = type(self).validate_config(state["config"])
