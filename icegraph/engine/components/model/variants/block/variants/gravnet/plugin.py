# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Any, ClassVar

from torch import Tensor
from torch.nn import Module, Linear, ModuleDict, LeakyReLU
from torch_geometric.nn import GravNetConv

from ...plugin import BlockModel

from .config import Config

__all__ = ["GravNet"]


class GravNet(BlockModel[Config]):
    """GravNet model for graph-level predictions."""
    name: ClassVar[str] = "gravnet"
    version: ClassVar[int] = 1

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> Config:
        return Config(**config)

    def _build_out(self, in_channels: int, out_channels: int) -> Module:
        # load required params
        hidden_c = self.config.hidden_channels

        # set final layer to Linear
        return Linear(hidden_c, out_channels)

    def _build_blocks(self, in_channels: int, out_channels: int) -> list[ModuleDict]:
        # load required params
        hidden_c    = self.config.hidden_channels
        layers      = self.config.hidden_layers
        space_dims  = self.config.space_dimensions
        prop_dims   = self.config.propagate_dimensions
        num_nbrs    = self.config.num_neighbors

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

    def _build_activation(self) -> Module:
        return LeakyReLU(negative_slope=0.01)

    def _forward_block(self, tensor: Tensor, batch: Tensor, block: ModuleDict, activation: Module) -> Tensor:
        tensor = block["conv"](tensor, batch)

        # run activation and return
        return activation(block["linear"](tensor))
