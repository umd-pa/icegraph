# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import ClassVar

from icegraph.common.transforms import TransformSpace
from icegraph.statistics import StatisticService

from ..plugin import AffineNormalizer

__all__ = ["MinMax"]


class MinMax(AffineNormalizer):
    """Scale to [0, 1] range."""
    name: ClassVar[str] = "minmax"
    version: ClassVar[int] = 1

    def _build_offset(self, stats: StatisticService, space: TransformSpace, base: int, column_index: int) -> float:
        # offset is minimum (shift to 0)
        minimum = float(stats.get("min", space=space, base=base)[column_index])
        return minimum

    def _build_scale(self, stats: StatisticService, space: TransformSpace, base: int, column_index: int) -> float:
        # scale factor is 1/range
        range_ = float(stats.range(space=space, base=base)[column_index])
        return 1.0 / max(range_, 1e-12)
