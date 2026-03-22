# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import ClassVar

import torch
from torch import Tensor

from icegraph.types.transforms import TransformSpace
from icegraph.statistics import StatisticService

from ..plugin import AffineNormalizer

__all__ = ["UnitVariance"]


class UnitVariance(AffineNormalizer):
    """UnitVariance normalizer for scaling input features and target labels by inverse variance."""
    name: ClassVar[str] = "variance"
    version: ClassVar[int] = 1

    def _build_offset(self, stats: StatisticService, space: TransformSpace, base: int) -> Tensor:
        # offset is 0, no offset
        zeros = torch.as_tensor(
            [0] * len(stats.columns),
            dtype=torch.float32
        )
        return zeros

    def _build_scale(self, stats: StatisticService, space: TransformSpace, base: int) -> Tensor:
        # scale factor is inverse variance
        variance = torch.as_tensor(
            stats.variance(space=space, base=base),
            dtype=torch.float32
        )
        return variance.clamp_min(1e-12).reciprocal()
