# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

import numpy as np

from icegraph.types.transforms import TransformSpace
from icegraph.types.statistics import StatisticKind
from icegraph.types.common import ArrayF

from ..statistic import Statistic
from ..bundle import StatisticBundle

__all__ = ["TotalCount"]


class TotalCount(Statistic):
    name = StatisticKind.TOTAL_COUNT
    degree = 0
    spaces = (TransformSpace.LINEAR,)

    def _compute(self, array: ArrayF) -> ArrayF:
        return np.full(array.shape[1], float(array.shape[0]))

    @classmethod
    def _merge(cls, a: StatisticBundle, b: StatisticBundle, space: TransformSpace) -> ArrayF:
        # simple add for merge
        return a.get(cls.name).value(space) + b.get(cls.name).value(space)
