# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

import logging
from typing import Any, ClassVar

import torch
from torch import Tensor
from torch.nn import Module, ModuleList, ModuleDict, Linear, LeakyReLU
from torch_geometric.nn import GCNConv, global_mean_pool

from icegraph.common.tensors import SegmentedTensor
from icegraph.engine.components.model import Model

from .config import GCNConfig

__all__ = ["GCN"]

logger = logging.getLogger(__name__)


class GCN(Model[GCNConfig]):
    """Graph convolutional network for graph-level predictions."""

    name: ClassVar[str] = "gcn"
    version: ClassVar[int] = 1

    # make the type checker happy
    _blocks:     ModuleList
    _activation: Module
    _out:        Module

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> GCNConfig:
        return GCNConfig(**config)

    def on_attach(self) -> None:
        in_channels = self.in_channels
        out_channels = self.out_channels

        hidden_c = self.config.hidden_channels
        layers   = self.config.hidden_layers

        # final layer
        self._out = Linear(hidden_c, out_channels)

        # activation
        self._activation = LeakyReLU(negative_slope=0.01)

        # blocks
        blocks: list[ModuleDict] = []
        for layer in range(layers):
            blocks.append(ModuleDict({
                "conv": GCNConv(
                    in_channels=in_channels if layer == 0 else hidden_c,
                    out_channels=hidden_c,
                    add_self_loops=False
                ),
                "linear": Linear(hidden_c, hidden_c)
            }))

        self._blocks = ModuleList(blocks)

    def supports_amp(self) -> bool:
        # override to true to opt in to AMP
        return True

    def forward_pass(
        self,
        t: SegmentedTensor,
        /,
        edge_index: Tensor,
        edge_attr: Tensor,
        batch: Tensor | None,
        num_graphs: int | None
    ) -> Tensor:
        # we dont care about segments here
        data = t.data.to(torch.float32)

        if batch is None:
            logger.warning("no batch was provided to the model, assuming all are nodes of the same graph")
            # if batch is not provided (single-sample inference), create a dummy batch tensor
            # this assumes all nodes belong to a single graph
            batch = torch.zeros(data.size(0), dtype=torch.long, device=data.device)
            num_graphs = 1

        # GCNConv wants a 1d edge_weight, collapse edge_attr [E, ATTR] -> [E]
        edge_weight: Tensor | None = None
        if edge_attr.numel() > 0:
            if edge_attr.size(1) != 1:
                raise ValueError(
                    f"{type(self).__name__} expects edge_attr of shape [E, 1] to use as "
                    f"GCNConv edge weights, got {tuple(edge_attr.shape)}. GCNConv cannot "
                    f"consume multi-dimensional edge features."
                )
            edge_weight = edge_attr.squeeze(1).to(torch.float32)

        edge_index = edge_index.to(torch.long)

        # add self loops
        edge_index, edge_weight = self._add_self_loops(edge_index, edge_weight, data.size(0))

        # forward pass through each block
        for block in self._blocks:
            data = block["conv"](data, edge_index, edge_weight)  # pyright: ignore[reportIndexIssue]
            data = self._activation(block["linear"](data))  # pyright: ignore[reportIndexIssue]

        data = global_mean_pool(data.float(), batch, size=num_graphs)
        return self._out(data)

    def _add_self_loops(
            self,
            edge_index: Tensor,
            edge_weight: Tensor | None,
            num_nodes: int
    ) -> tuple[Tensor, Tensor | None]:
        """Append one self loop per node, as GCNConv would with its default add_self_loops."""
        loops = torch.arange(num_nodes, dtype=torch.long, device=edge_index.device)
        edge_index = torch.cat([edge_index, torch.stack([loops, loops])], dim=1)

        # GCNConv weights an added loop 1
        if edge_weight is not None:
            edge_weight = torch.cat([edge_weight, edge_weight.new_ones(num_nodes)])

        return edge_index, edge_weight