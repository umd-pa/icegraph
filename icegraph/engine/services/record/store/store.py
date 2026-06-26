# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from abc import abstractmethod
from typing import  Iterator, TypeVar, overload
from functools import cached_property

from icegraph.common.plugins import Plugin
from icegraph.common.record import Attributes, GlobalAttributes, Record

from .types import StoreContext

__all__ = ["Store"]


C = TypeVar("C")


class Store(Plugin[C, StoreContext]):
    """Provides access to dataset files."""

    @overload
    def __getitem__(self, index: int) -> Record: ...
    @overload
    def __getitem__(self, index: slice) -> list[Record]: ...
    @overload
    def __getitem__(self, index: int | slice) -> Record | list[Record]: ...

    @abstractmethod
    def __getitem__(self, index: int | slice) -> Record | list[Record]:
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...

    @cached_property
    def global_attrs(self) -> GlobalAttributes:
        return GlobalAttributes.from_attrs(self.attrs, ignore_checksum=self._ctx.ignore_checksum)

    @property
    @abstractmethod
    def attrs(self) -> Iterator[Attributes]:
        """Iterate over all shard attributes in the dataset in a deterministic order."""
        ...

    @abstractmethod
    def close(self) -> None:
        ...
