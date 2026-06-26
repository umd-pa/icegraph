# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Protocol, Any, overload, Literal, TypeAlias
from collections.abc import Iterator

from icegraph.typing.common import ArrayF

__all__ = ["StatisticBundleStruct", "StatisticStruct"]


StatisticStruct: TypeAlias = dict[str, ArrayF]


class StatisticBundleStruct(Protocol):
    @overload
    def __getitem__(self, item: Literal["stats"]) -> dict[str, StatisticStruct]: ...
    @overload
    def __getitem__(self, item: str) -> Any: ...
    def __getitem__(self, item: str) -> Any: ...

    def __iter__(self) -> Iterator[str]: ...
    def __len__(self) -> int: ...