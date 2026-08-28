# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import ClassVar

from icegraph.common.transforms import TransformSpace
from icegraph.statistics import StatisticService

from ...plugin import AffineNormalizer

__all__ = ["UnitVariance"]


class UnitVariance(AffineNormalizer):
    """Scale by inverse standard deviation without centering."""
    name: ClassVar[str] = "unit-variance"
    version: ClassVar[int] = 1

    def _build_offset(self, stats: StatisticService, space: TransformSpace, base: int, column_index: int) -> float:
        # offset is 0, no offset
        return 0.0

    def _build_scale(self, stats: StatisticService, space: TransformSpace, base: int, column_index: int) -> float:
        # scale factor is reciprocal of std
        std = float(stats.std(space=space, base=base)[column_index])
        return 1.0 / max(std, 1e-12)
