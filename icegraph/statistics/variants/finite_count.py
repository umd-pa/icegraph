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

__all__ = ["FiniteCount"]


class FiniteCount(Statistic):
    name: ClassVar[str] = "finite_count"
    degree = 0
    spaces = (TransformSpace.LINEAR,)

    def _compute(self, array: ArrayF) -> ArrayF:
        # per-column count of finite values (excludes NaN and +-inf)
        return np.isfinite(array).sum(axis=0).astype(float)

    @classmethod
    def _merge(cls, a: StatisticBundle, b: StatisticBundle, space: TransformSpace) -> ArrayF:
        # simple add for merge
        return a.get(cls.name).value(space) + b.get(cls.name).value(space)
