# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import ClassVar

import torch
from torch import Tensor

from icegraph.types.transforms import TransformSpace
from icegraph.statistics import StatisticService

from ..affine import AffineNormalizer

__all__ = ["MinMax"]


class MinMax(AffineNormalizer):
    """MinMax normalizer for scaling input features and target labels to [0, 1] range."""
    name: ClassVar[str] = "minmax"

    def _build_offset(self, stats: StatisticService, space: TransformSpace, base: int) -> Tensor:
        # offset is minimum (shift to 0)
        minimum = torch.as_tensor(
            stats.get("min", space=space, base=base),
            dtype=torch.float32
        )
        return minimum

    def _build_scale(self, stats: StatisticService, space: TransformSpace, base: int) -> Tensor:
        # scale factor is 1/range
        range_ = torch.as_tensor(
            stats.range(space=space, base=base),
            dtype=torch.float32
        )
        return range_.clamp_min(1e-12).reciprocal()
