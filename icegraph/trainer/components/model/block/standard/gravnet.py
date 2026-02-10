# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Any, ClassVar

from torch import Tensor
from torch.nn import Module, Linear, ModuleDict, LeakyReLU
from torch_geometric.nn import GravNetConv

from ..block import BlockModel
from ..config import GravNetConfig

__all__ = ["GravNet"]


class GravNet(BlockModel[GravNetConfig]):
    """
    GravNet model for graph-level predictions.

    The network applies successive GravNet convolutional layers to learn
    node embeddings, each followed by a linear transformation and ReLU activation.
    Finally, it aggregates node embeddings with global mean pooling and computes
    graph-level outputs via a final linear layer.
    """
    name: ClassVar[str] = "gravnet"

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> GravNetConfig:
        return GravNetConfig(**config)

    def _build_out(self, config: GravNetConfig, in_channels: int, out_channels: int) -> Module:
        # load required params
        hidden_c = config.hidden_channels

        # set final layer to Linear
        return Linear(hidden_c, out_channels)

    def _build_blocks(self, config: GravNetConfig, in_channels: int, out_channels: int) -> list[ModuleDict]:
        # load required params
        hidden_c    = config.hidden_channels
        layers      = config.hidden_layers
        space_dims  = config.space_dimensions
        prop_dims   = config.propagate_dimensions
        num_nbrs    = config.num_neighbors

        modules: list[ModuleDict] = []
        for layer in range(layers):
            modules.append(ModuleDict({
                "conv": GravNetConv(
                    in_channels=in_channels if layer == 0 else hidden_c,
                    out_channels=hidden_c,
                    space_dimensions=space_dims,
                    propagate_dimensions=prop_dims,
                    k=num_nbrs
                ),
                "linear": Linear(hidden_c, hidden_c)
            }))

        return modules

    def _build_activation(self, config: GravNetConfig) -> Module:
        return LeakyReLU(negative_slope=0.01)

    def _forward_block(self, tensor: Tensor, batch: Tensor, block: ModuleDict, activation: Module) -> Tensor:
        tensor = block["conv"](tensor, batch)

        # run activation and return
        return activation(block["linear"](tensor))
