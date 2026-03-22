# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from abc import abstractmethod
from typing import Any, Sequence, cast, TypeVar

import torch
from torch import Tensor
from torch.nn import Module, ModuleList, ModuleDict
import torch_geometric

from icegraph.trainer.components.model import Model

__all__ = ["BlockModel"]


C = TypeVar("C")


class BlockModel(Model[C]):

    # make the type checker happy
    _blocks:        ModuleList | None
    _activation:    Module | None
    _out:           Module | None

    def build(self) -> None:
        # cache for activation and blocks
        self._blocks        = None
        self._activation    = None
        self._out           = None

    def on_attach(self) -> None:
        # get strategy service
        strategy = self._ctx.services.require("strategy", required_by=type(self))

        # eagerly build out, blocks, and activation
        self._out           = self._build_out(strategy.in_channels, strategy.out_channels)
        self._blocks        = ModuleList(self._build_blocks(strategy.in_channels, strategy.out_channels))
        self._activation    = self._build_activation()

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
        return self._out

    @property
    def blocks(self) -> Sequence[ModuleDict]:
        return cast(Sequence[ModuleDict], cast(object, self._blocks))

    @property
    def activation(self) -> Module:
        return self._activation

    @abstractmethod
    def _build_blocks(self, in_channels: int, out_channels: int) -> list[ModuleDict]:
        ...

    @abstractmethod
    def _build_activation(self) -> Module:
        ...

    @abstractmethod
    def _build_out(self, in_channels: int, out_channels: int) -> Module:
        ...

    @abstractmethod
    def _forward_block(self, tensor: Tensor, batch: Tensor, block: ModuleDict, activation: Module) -> Tensor:
        ...

    def get_extra_state(self) -> dict[str, Any]:
        # Must be picklable
        return {"config": self.config.model_dump(mode="json")}

    def set_extra_state(self, state: dict[str, Any]) -> None:
        self.config = type(self).validate_config(state["config"])
