# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

import logging
from typing import Any, ClassVar

import torch
from torch import Tensor
from torch.nn import Module, ModuleList, Linear, LeakyReLU, Sequential
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import GINEConv, global_mean_pool

from icegraph.common.tensors import SegmentedTensor
from icegraph.engine.components.model import Model

from .config import GINEConfig

__all__ = ["GINE"]

logger = logging.getLogger(__name__)


class GINE(Model[GINEConfig]):
    """Graph isomorphism network with edge features for graph-level predictions."""

    name: ClassVar[str] = "gine"
    version: ClassVar[int] = 1

    # make the type checker happy
    _convs:      ModuleList
    _activation: Module
    _out:        Module

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> GINEConfig:
        return GINEConfig(**config)

    def on_attach(self) -> None:
        in_channels = self.in_channels
        out_channels = self.out_channels

        hidden_c = self.config.hidden_channels
        layers   = self.config.hidden_layers

        # final layer
        self._out = Linear(hidden_c, out_channels)

        # activation
        self._activation = LeakyReLU(negative_slope=0.01)

        # convs, each one updates nodes with a two layer MLP
        convs: list[GINEConv] = []
        for layer in range(layers):
            mlp = Sequential(
                Linear(in_channels if layer == 0 else hidden_c, hidden_c),
                LeakyReLU(negative_slope=0.01),
                Linear(hidden_c, hidden_c)
            )
            convs.append(GINEConv(
                nn=mlp,
                train_eps=self.config.train_eps,
                # GINEConv projects edge_attr to the input width of mlp with its own linear layer
                edge_dim=self.config.edge_channels
            ))

        self._convs = ModuleList(convs)

    def supports_amp(self) -> bool:
        # override to true to opt in to AMP
        return True

    def _layer(self, conv: Module, data: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        return self._activation(conv(data, edge_index, edge_attr))

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
        data: Tensor = t.data.to(torch.float32)

        if batch is None:
            logger.warning("no batch was provided to the model, assuming all are nodes of the same graph")
            # if batch is not provided (single-sample inference), create a dummy batch tensor
            # this assumes all nodes belong to a single graph
            batch = torch.zeros(data.size(0), dtype=torch.long, device=data.device)
            num_graphs = 1

        # the edge projection in GINEConv is sized at attach time, so the width must match the config
        if edge_attr.size(1) != self.config.edge_channels:
            raise ValueError(
                f"{type(self).__name__} expects edge_attr of shape [E, {self.config.edge_channels}] "
                f"as set by 'edge_channels', got {tuple(edge_attr.shape)}."
            )

        # cast once here, not once per layer
        edge_index = edge_index.to(torch.long)
        edge_attr = edge_attr.to(torch.float32)

        # checkpointing only helps when autograd is saving activations for a backward pass
        use_checkpoint = self.config.checkpoint and self.training and torch.is_grad_enabled()

        # forward pass through each conv
        for conv in self._convs:
            if use_checkpoint:
                # keep only the layer input, and recompute the per edge messages during backward
                data = checkpoint(self._layer, conv, data, edge_index, edge_attr, use_reentrant=False)  # pyright: ignore[reportAssignmentType]
            else:
                data = self._layer(conv, data, edge_index, edge_attr)

        data = global_mean_pool(data.float(), batch, size=num_graphs)
        return self._out(data)
