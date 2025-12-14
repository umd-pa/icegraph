# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import Dict, Optional, TypeAlias, Any

import torch
import torch.nn as nn
from torch_geometric.nn import GravNetConv, global_mean_pool

__all__ = ["GravNet"]

OptInt: TypeAlias = Optional[int]


class GravNet(nn.Module):
    """
    GravNet model for graph-level predictions.

    The network applies successive GravNet convolutional layers to learn
    node embeddings, each followed by a linear transformation and ReLU activation.
    Finally, it aggregates node embeddings with global mean pooling and computes
    graph-level outputs via a final linear layer.

    Args:
        in_channels (int): Number of input features per node.
        out_channels (int): Dimension of the final graph-level output.
    """
    def __init__(
            self, in_channels: OptInt, hidden_channels: OptInt, out_channels: OptInt, layers: OptInt, k: OptInt
    ) -> None:
        super().__init__()

        self._params = {
            "in_channels"       : in_channels,
            "hidden_channels"   : hidden_channels,
            "out_channels"      : out_channels,
            "layers"            : layers,
            "k"                 : k
        }

        self.blocks:        Optional[nn.ModuleList]     = None
        self.activation:    Optional[nn.Module]         = None
        self.out:           Optional[nn.Module]         = None

    def init_model(self) -> None:
        if not all(self._params.values()):
            raise ValueError("Missing one or more model configuration parameters.")

        self.blocks = nn.ModuleList()
        self.activation = nn.LeakyReLU(negative_slope=0.01)

        # grab params from dict
        in_channels         = self._params["in_channels"]
        hidden_channels     = self._params["hidden_channels"]
        out_channels        = self._params["out_channels"]
        layers              = self._params["layers"]
        k                   = self._params["k"]

        for i in range(layers):
            conv = GravNetConv(
                in_channels=in_channels if i == 0 else hidden_channels,
                out_channels=hidden_channels,
                space_dimensions=4,
                propagate_dimensions=12,
                k=k
            )
            lin = nn.Linear(hidden_channels, hidden_channels)

            self.blocks.append(nn.ModuleDict({
                "gravnetconv": conv,
                "linear": lin
            }))

        self.out = nn.Linear(hidden_channels, out_channels)

        # remove upward bias
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor, batch: Optional[torch.LongTensor] = None) -> torch.Tensor:
        """
        Forward pass through the GravNet architecture.

        Args:
            x (torch.Tensor): Node feature matrix with shape [num_nodes, in_channels].
            batch (torch.LongTensor, optional): Batch vector which maps each node to its graph in the batch.

        Returns:
            Tensor: Graph-level predictions with shape [num_graphs, out_channels].
        """
        if any([m is None for m in [self.blocks, self.activation, self.out]]):
            raise ValueError(
                "Model has not been initialized (or only partially initialized), please call `init_model()`."
            )

        if batch is None:
            # If batch is not provided (single-sample inference), create a dummy batch tensor.
            # this assumes all nodes belong to a single graph
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # pass through each block in the model
        for block in self.blocks:
            x = block["gravnetconv"](x, batch)

            # run activation
            x = self.activation(block["linear"](x))

        x = global_mean_pool(x, batch)
        return self.out(x)

    def get_extra_state(self) -> Dict[str, Any]:
        # Must be picklable
        return self._params

    def set_extra_state(self, state: Dict[str, Any]) -> None:
        self._params = state
