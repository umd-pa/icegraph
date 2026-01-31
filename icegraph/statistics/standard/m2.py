# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

import numpy as np

from icegraph.types.transforms import TransformSpace
from icegraph.types.statistics import StatisticKind
from icegraph.types.common import ArrayF

from ..statistic import Statistic
from ..bundle import StatisticBundle

__all__ = ["WelfordM2"]


class WelfordM2(Statistic):
    """
    Welford M2 provides a numerically stable method of calculating variance. Allows computation of
    variance and related metrics in a single pass.
    """
    name = StatisticKind.M2
    degree = 2

    def _compute(self, array: ArrayF) -> ArrayF:
        # compute m2, see https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        mean = np.nanmean(array, axis=0)
        diff = array - mean
        return np.nansum(diff * diff, axis=0)

    @classmethod
    def _merge(cls, a: StatisticBundle, b: StatisticBundle, space: TransformSpace) -> ArrayF:
        # chan-golub-leveque merge, see http://i.stanford.edu/pub/cstr/reports/cs/tr/79/773/CS-TR-79-773.pdf
        linear = TransformSpace.LINEAR

        # space is guaranteed to be one of log, asinh or linear, so we can skip checks
        # linear and asinh is computed over finite non-nan values
        count_a = a.get(StatisticKind.FINITE_COUNT).value(linear)
        count_b = b.get(StatisticKind.FINITE_COUNT).value(linear)

        if space == TransformSpace.LOG:
            # log is computed over positive finite non-nan non-zero values only
            count_a = a.get(StatisticKind.POSITIVE_COUNT).value(linear)
            count_b = b.get(StatisticKind.POSITIVE_COUNT).value(linear)

        # get mean from each bundle
        mean_a = a.get(StatisticKind.MEAN).value(space)
        mean_b = b.get(StatisticKind.MEAN).value(space)

        # get m2 from each bundle
        m2_a = a.get(cls.name).value(space)
        m2_b = b.get(cls.name).value(space)

        denom = count_a + count_b

        # compute delta
        delta = mean_b - mean_a

        with np.errstate(divide="ignore", invalid="ignore"):
            # compute correction term
            correction = (delta * delta) * (count_a * count_b / denom)
            m2 = m2_a + m2_b + np.where(denom > 0, correction, 0.0)

        # if denom==0 -> nan
        return np.where(denom > 0, m2, np.nan)