# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["build_label_index_map", "flatten"]


def build_label_index_map(indices: Tensor) -> Tensor:
        # build label index mapping
        batch_count, label_count, _ = indices.shape
        label_index_map = (torch
           .arange(label_count, device=indices.device, dtype=torch.long)  # build list [0, 1, ..., label_count - 1]
           .view(1, label_count, 1)  # xfrm to [[0, 1, ..., label_count - 1]]
           .expand(batch_count, label_count, 1)  # duplicate batch_count times (duplicated logically, no mem alloc)
        )

        return label_index_map

def flatten(indices: Tensor, index_map: Tensor, bins: Tensor):
    # flatten to sparse array
    cell_count = torch.prod(bins)  # okay to do here, very cheap (only 2 items)
    flat = index_map * cell_count + indices[..., 1] * bins[0] + indices[..., 0]

    # return flattened tensor
    return flat
