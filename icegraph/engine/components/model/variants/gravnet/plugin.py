# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

import logging
from typing import Any, ClassVar

import torch
from torch import Tensor
from torch.nn import Module, ModuleList, ModuleDict, Linear, LeakyReLU
from torch_geometric.nn import GravNetConv, global_mean_pool

from icegraph.common.tensors import SegmentedTensor
from icegraph.engine.components.model import Model

from .config import GravNetConfig

__all__ = ["GravNet"]

logger = logging.getLogger(__name__)


class GravNet(Model[GravNetConfig]):
    """GravNet model for graph-level predictions."""

    name: ClassVar[str] = "gravnet"
    version: ClassVar[int] = 1

    # make the type checker happy
    _blocks:     ModuleList
    _activation: Module
    _out:        Module

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> GravNetConfig:
        return GravNetConfig(**config)

    def on_attach(self) -> None:
        in_channels = self.in_channels
        out_channels = self.out_channels

        hidden_c   = self.config.hidden_channels
        layers     = self.config.hidden_layers
        space_dims = self.config.space_dimensions
        prop_dims  = self.config.propagate_dimensions
        num_nbrs   = self.config.num_neighbors

        # final layer
        self._out = Linear(hidden_c, out_channels)

        # activation
        self._activation = LeakyReLU(negative_slope=0.01)

        # blocks
        blocks: list[ModuleDict] = []
        for layer in range(layers):
            blocks.append(ModuleDict({
                "conv": GravNetConv(
                    in_channels=in_channels if layer == 0 else hidden_c,
                    out_channels=hidden_c,
                    space_dimensions=space_dims,
                    propagate_dimensions=prop_dims,
                    k=num_nbrs
                ),
                "linear": Linear(hidden_c, hidden_c)
            }))

        self._blocks = ModuleList(blocks)

    def forward_pass(
            self,
            t: SegmentedTensor,
            /,
            edge_index: Tensor,
            edge_attr: Tensor,
            batch: Tensor | None
    ) -> Tensor:
        # we dont care about segments here
        data = t.data.to(torch.float32)

        if batch is None:
            logger.warning("no batch was provided to the model, assuming all are nodes of the same graph")
            # if batch is not provided (single-sample inference), create a dummy batch tensor
            # this assumes all nodes belong to a single graph
            batch = torch.zeros(data.size(0), dtype=torch.long, device=data.device)

        # forward pass through each block
        for block in self._blocks:
            data = block["conv"](data, batch)  # pyright: ignore[reportIndexIssue]
            data = self._activation(block["linear"](data))  # pyright: ignore[reportIndexIssue]

        data = global_mean_pool(data, batch)
        return self._out(data)