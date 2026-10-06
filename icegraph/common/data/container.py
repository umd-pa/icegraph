# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Self, Callable, Mapping

from torch_geometric.data import Batch
from jaxtyping import Int, Float
import torch

from icegraph.common.data import DataRole, ColumnarRole

from ..tensors import SegmentedTensor, SegmentLayout

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["GraphBatch"]


@dataclass(frozen=True, eq=False)
class GraphBatch:
    """Batched graphs, assembled columnar by the decode service."""
    # features (jagged, store flattened, in the vast majority of cases each segment is unity)
    features:       SegmentedTensor

    # truth (both jagged, but stored flattened)
    targets:        SegmentedTensor
    auxiliary:      SegmentedTensor

    # weights
    weights:        Tensor

    # batch vector and per-graph node offsets
    batch:          Tensor
    ptr:            Tensor

    @property
    def num_graphs(self) -> int:
        """Graphs in the batch, from the offsets' shape so nothing waits on the device."""
        return self.ptr.numel() - 1

    @property
    def num_nodes(self) -> int:
        """Nodes across every graph in the batch, from the batch vector's shape."""
        return self.batch.numel()

    def apply(self, fn: Callable[[Tensor | SegmentedTensor], Tensor | SegmentedTensor], /) -> Self:
        """Apply a function to all tensors and segmented tensors in the batch"""
        for f in fields(self):
            tensor: Tensor | SegmentedTensor = getattr(self, f.name)

            # apply fn to tensor
            mutated = fn(tensor)

            # ensure mutation does not modify type
            if type(tensor) is not type(mutated):
                raise TypeError(
                    f"{type(self).__name__}.apply: fn must not modify type, expected {type(tensor)}, got {type(mutated)}."
                )

            # skip frozen check, reassign each
            object.__setattr__(self, f.name, mutated)

        return self

    def detach(self) -> Self:
        """Detach all tensors in the batch."""
        return self.apply(lambda t: t.detach())

    def pin_memory(self) -> Self:
        """Run pin_memory() on all tensors in the batch."""
        return self.apply(lambda t: t.pin_memory())

    def to(
        self,
        device: torch.device | str | int | None = None, *,
        non_blocking: bool = False,
    ) -> Self:
        """Move all tensors in the batch to device. Use to_dtype() for data type casts."""
        return self.apply(lambda t: t.to(device, non_blocking=non_blocking))

    def to_dtype(
        self,
        mapping: Mapping[DataRole, torch.dtype]
    ) -> Self:
        """Cast any role to a new data type via a mapping."""
        # only need to remap what was passed
        for role, dtype in mapping.items():
            tensor: Tensor | SegmentedTensor = getattr(self, role.value)

            # skip frozen check, cast each tensor
            object.__setattr__(self, role.value, tensor.to(dtype=dtype))

        return self
