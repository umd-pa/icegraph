# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import ClassVar

from icegraph.common.transforms import TransformSpace
from icegraph.statistics import StatisticService

from ..plugin import AffineNormalizer

__all__ = ["MeanCentering"]


class MeanCentering(AffineNormalizer):
    """Shift by the mean without scaling."""
    name: ClassVar[str] = "mean-centering"
    version: ClassVar[int] = 1

    def _build_offset(self, stats: StatisticService, space: TransformSpace, base: int, column_index: int) -> float:
        # offset is simply the mean
        mean = float(stats.get("mean", space=space, base=base)[column_index])
        return mean

    def _build_scale(self, stats: StatisticService, space: TransformSpace, base: int, column_index: int) -> float:
        # scale factor is 1 (no scaling)
        return 1.0
