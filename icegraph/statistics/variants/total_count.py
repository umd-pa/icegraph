# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import ClassVar, TYPE_CHECKING
from typing_extensions import override

import numpy as np

from icegraph.common.transforms import TransformSpace
from icegraph.typing.common import ArrayF

from ..statistic import Statistic

if TYPE_CHECKING:
    from ..bundle import StatisticBundle

__all__ = ["TotalCount"]


class TotalCount(Statistic):
    name: ClassVar[str] = "total_count"
    degree = 0
    spaces = (TransformSpace.LINEAR,)

    @override
    def _compute(self, array: ArrayF) -> ArrayF:
        return np.full(array.shape[1], float(array.shape[0]))

    @classmethod
    @override
    def _merge(cls, a: StatisticBundle, b: StatisticBundle, space: TransformSpace) -> ArrayF:
        # simple add for merge
        return a.get(cls.name).value(space) + b.get(cls.name).value(space)
