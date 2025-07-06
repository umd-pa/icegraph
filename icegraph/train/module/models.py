# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

import torch
import torch.nn as nn
from torch_geometric.nn import GravNetConv, global_mean_pool

__all__ = ["GravNetConv"]


class GravNetModel(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()

        self.gravnet1 = GravNetConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            space_dimensions=4,
            propagate_dimensions=12,
            k=16
        )
        self.lin1 = nn.Linear(hidden_channels, hidden_channels)

        self.gravnet2 = GravNetConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            space_dimensions=4,
            propagate_dimensions=12,
            k=16
        )
        self.lin2 = nn.Linear(hidden_channels, hidden_channels)

        self.out = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, batch):
        x = self.gravnet1(x, batch)
        x = torch.relu(self.lin1(x))

        x = self.gravnet2(x, batch)
        x = torch.relu(self.lin2(x))

        x = global_mean_pool(x, batch)  # [num_graphs, hidden_channels]
        return self.out(x)              # [num_graphs, out_channels]