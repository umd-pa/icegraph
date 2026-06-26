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

__all__ = ["NANCount"]


class NANCount(Statistic):
    name: ClassVar[str] = "nan_count"
    degree = 0
    spaces = (TransformSpace.LINEAR,)

    @override
    def _compute(self, array: ArrayF) -> ArrayF:
        return np.isnan(array).sum(axis=0).astype(float)

    @classmethod
    @override
    def _merge(cls, a: StatisticBundle, b: StatisticBundle, space: TransformSpace) -> ArrayF:
        # simple add for merge
        return a.get(cls.name).value(space) + b.get(cls.name).value(space)
