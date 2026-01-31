# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from icegraph.types.factory import ModuleFactory
from icegraph.types.statistics import StatisticKind, StatisticStruct

from .statistic import Statistic
from .standard import *

__all__ = ["StatisticFactory"]


class StatisticFactory(ModuleFactory[StatisticKind, Statistic]):

    @classmethod
    def from_struct(cls, name: StatisticKind, struct: StatisticStruct) -> Statistic:
        """Instantiate a registered statistic from a struct."""
        spec = cls._typed_registry()[name]
        return spec.from_struct(struct)


for stat in [Minimum, Maximum, Mean, WelfordM2, PositiveCount, NANCount, FiniteCount, ZeroCount, TotalCount]:
    StatisticFactory.register(stat)


