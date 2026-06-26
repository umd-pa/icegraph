# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Protocol

from torch import Tensor

from icegraph.common.tensors import SegmentedTensor

__all__ = ["BoundModel"]


class BoundModel(Protocol):
    def __call__(self, t: SegmentedTensor, /, batch: Tensor | None = None) -> SegmentedTensor: ...
