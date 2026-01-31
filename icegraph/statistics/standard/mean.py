# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

import numpy as np

from icegraph.types.transforms import TransformSpace
from icegraph.types.statistics import StatisticKind
from icegraph.types.common import ArrayF

from ..statistic import Statistic
from ..bundle import StatisticBundle

__all__ = ["Mean"]

class Mean(Statistic):
    name = StatisticKind.MEAN
    degree = 1

    def _compute(self, array: ArrayF) -> ArrayF:
        # take mean, ignore nans
        return np.nanmean(array, axis=0)

    @classmethod
    def _merge(cls, a: StatisticBundle, b: StatisticBundle, space: TransformSpace) -> ArrayF:
        linear = TransformSpace.LINEAR

        # space is guaranteed to be one of log, asinh or linear, so we can skip checks
        count_a = a.get(StatisticKind.FINITE_COUNT).value(linear)
        count_b = b.get(StatisticKind.FINITE_COUNT).value(linear)

        if space == TransformSpace.LOG:
            # log is computed over positive finite values only
            count_a = a.get(StatisticKind.POSITIVE_COUNT).value(linear)
            count_b = b.get(StatisticKind.POSITIVE_COUNT).value(linear)

        # get mean from each bundle
        mean_a = a.get(cls.name).value(space)
        mean_b = b.get(cls.name).value(space)

        denom = count_a + count_b

        # avoid NaN * 0 poisoning by zeroing the mean where count==0
        a_term = np.where(count_a > 0, mean_a * count_a, 0.0)
        b_term = np.where(count_b > 0, mean_b * count_b, 0.0)

        with np.errstate(divide="ignore", invalid="ignore"):
            mean = (a_term + b_term) / denom

        # if denom==0 -> nan
        return np.where(denom > 0, mean, np.nan)
