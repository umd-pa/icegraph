# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import TypeAlias

from icegraph.typing.common import ArrayF

__all__ = ["StatisticBundleStruct", "StatisticStruct"]


StatisticStruct:        TypeAlias = dict[str, ArrayF]
StatisticBundleStruct:  TypeAlias = dict[str, list[str] | dict[str, StatisticStruct]]
