# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from icegraph.common.factory import Factory

from . import variants
from .statistic import Statistic
from .types import StatisticStruct

__all__ = ["StatisticFactory"]


class StatisticFactory(Factory[Statistic]):

    @classmethod
    def from_struct(cls, name: str, struct: StatisticStruct) -> Statistic:
        """Instantiate a registered statistic from a struct."""
        spec = cls._typed_registry()[name]
        return spec.from_struct(struct)


for name_ in variants.__all__:
    StatisticFactory.register(getattr(variants, name_))


