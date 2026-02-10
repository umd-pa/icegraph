# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import ClassVar

import numpy as np

from icegraph.types.transforms import TransformSpace
from icegraph.types.common import ArrayF

from ..statistic import Statistic
from ..bundle import StatisticBundle

__all__ = ["ZeroCount"]


class ZeroCount(Statistic):
    name: ClassVar[str] = "zero_count"
    degree = 0
    spaces = (TransformSpace.LINEAR,)

    def _compute(self, array: ArrayF) -> ArrayF:
        # per-column count of finite zeros
        return (np.isfinite(array) & (array == 0.0)).sum(axis=0).astype(float)

    @classmethod
    def _merge(cls, a: StatisticBundle, b: StatisticBundle, space: TransformSpace) -> ArrayF:
        # simple add for merge
        return a.get(cls.name).value(space) + b.get(cls.name).value(space)
