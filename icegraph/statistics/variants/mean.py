# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import ClassVar, TYPE_CHECKING

import numpy as np

from icegraph.common.transforms import TransformSpace
from icegraph.typing.common import ArrayF

from ..statistic import Statistic

if TYPE_CHECKING:
    from ..bundle import StatisticBundle

__all__ = ["Mean"]

class Mean(Statistic):
    name: ClassVar[str] = "mean"
    degree = 1

    def _compute(self, array: ArrayF) -> ArrayF:
        # take mean, ignore nans
        return np.nanmean(array, axis=0)

    @classmethod
    def _merge(cls, a: StatisticBundle, b: StatisticBundle, space: TransformSpace) -> ArrayF:
        linear = TransformSpace.LINEAR

        # space is guaranteed to be one of log, asinh or linear, so we can skip checks
        count_a = a.get("finite_count").value(linear)
        count_b = b.get("finite_count").value(linear)

        if space == TransformSpace.LOG:
            # log is computed over positive finite values only
            count_a = a.get("positive_count").value(linear)
            count_b = b.get("positive_count").value(linear)

        # get mean from each bundle
        mean_a = a.get(cls.name).value(space)
        mean_b = b.get(cls.name).value(space)

        denom = count_a + count_b

        # avoid NaN * 0 poisoning by zeroing the mean where count==0
        num = (
            np.where(count_a > 0, mean_a * count_a, 0.0)
            + np.where(count_b > 0, mean_b * count_b, 0.0)
        )

        mean = np.divide(num, denom, out=np.full_like(num, np.nan, dtype=float), where=denom > 0)
        return mean
