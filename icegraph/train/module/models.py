# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

import torch
import torch.nn as nn
from torch_geometric.nn import GravNetConv, global_mean_pool

from icegraph.config import IGConfig

__all__ = ["GravNet"]


class GravNet(nn.Module):
    """
    GravNet model for graph-level predictions.

    The network applies two successive GravNet convolutional layers to learn
    node embeddings, each followed by a linear transformation and ReLU activation.
    Finally, it aggregates node embeddings with global mean pooling and computes
    graph-level outputs via a final linear layer.

    Args:
        in_channels (int): Number of input features per node.
        out_channels (int): Dimension of the final graph-level output.
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, layers: int=2) -> None:
        super().__init__()

        self._config = IGConfig.get()
        k = self._config.user_config.training.trainer_params.num_nbrs

        self.blocks = nn.ModuleList()

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

    def forward(self, x: torch.Tensor, batch: torch.LongTensor):
        """
        Forward pass through the GravNet architecture.

        Args:
            x (torch.Tensor): Node feature matrix with shape [num_nodes, in_channels].
            batch (torch.LongTensor): Batch vector which maps each node to its graph in the batch.

        Returns:
            Tensor: Graph-level predictions with shape [num_graphs, out_channels].
        """
        for block in self.blocks:
            x = block["gravnetconv"](x, batch)
            x = torch.relu(block["linear"](x))

        x = global_mean_pool(x, batch)
        return self.out(x)