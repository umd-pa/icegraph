# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import ClassVar

import numpy as np

from icegraph.types.transforms import TransformSpace
from icegraph.types.common import ArrayF

from ..statistic import Statistic
from ..bundle import StatisticBundle

__all__ = ["Maximum"]


class Maximum(Statistic):
    name: ClassVar[str] = "max"
    degree = 1

    def _compute(self, array: ArrayF) -> ArrayF:
        # take maximum, ignore nans
        return np.nanmax(array, axis=0)

    @classmethod
    def _merge(cls, a: StatisticBundle, b: StatisticBundle, space: TransformSpace) -> ArrayF:
        # take elementwise max, ignore nans
        return np.fmax(a.get(cls.name).value(space), b.get(cls.name).value(space))
