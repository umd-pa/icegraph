# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import ClassVar, TYPE_CHECKING
from typing_extensions import override
import warnings

import numpy as np

from icegraph.common.transforms import TransformSpace
from icegraph.typing.common import ArrayF

from ..statistic import Statistic

if TYPE_CHECKING:
    from ..bundle import StatisticBundle

__all__ = ["WelfordM2"]


class WelfordM2(Statistic):
    """
    Welford M2 provides a numerically stable method of calculating variance. Allows computation of
    variance and related metrics in a single pass.
    """
    name: ClassVar[str] = "m2"
    degree = 2

    @override
    def _compute(self, array: ArrayF) -> ArrayF:
        # compute m2, see https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Mean of empty slice")
            mean = np.nanmean(array, axis=0)

        diff = array - mean
        return np.nansum(diff * diff, axis=0)

    @classmethod
    @override
    def _merge(cls, a: StatisticBundle, b: StatisticBundle, space: TransformSpace) -> ArrayF:
        # chan-golub-leveque merge, see http://i.stanford.edu/pub/cstr/reports/cs/tr/79/773/CS-TR-79-773.pdf
        linear = TransformSpace.LINEAR

        # space is guaranteed to be one of log, asinh or linear, so we can skip checks
        # linear and asinh is computed over finite non-nan values
        count_a = a.get("finite_count").value(linear)
        count_b = b.get("finite_count").value(linear)

        if space == TransformSpace.LOG:
            # log is computed over positive finite non-nan non-zero values only
            count_a = a.get("positive_count").value(linear)
            count_b = b.get("positive_count").value(linear)

        # get mean from each bundle
        mean_a = a.get("mean").value(space)
        mean_b = b.get("mean").value(space)

        # get m2 from each bundle
        m2_a = a.get(cls.name).value(space)
        m2_b = b.get(cls.name).value(space)

        denom = count_a + count_b

        # compute delta
        delta = mean_b - mean_a

        # compute merges m2
        ratio = np.divide(count_a * count_b, denom, out=np.zeros_like(denom, dtype=float), where=denom > 0)
        correction = np.where(ratio > 0, (delta * delta) * ratio, 0.0)
        m2 = m2_a + m2_b + correction

        # if denom==0 -> nan
        return np.where(denom > 0, m2, np.nan)