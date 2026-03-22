# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from abc import abstractmethod
from typing import  Iterator, Any, TypeVar

from icegraph.types.files import Source
from icegraph.types.plugins import Plugin

from ..types import Attributes, GlobalAttributes

from .types import StoreContext

__all__ = ["Store"]


C = TypeVar("C")


class Store(Plugin[C, StoreContext]):
    """Provides access to dataset files."""

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate through all records."""
        for i in range(len(self)):
            yield self[i]

    @abstractmethod
    def __getitem__(self, index: int | slice) -> dict[str, Any] | list[dict[str, Any]]:
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...

    @property
    @abstractmethod
    def global_attrs(self) -> GlobalAttributes:
        ...

    @property
    @abstractmethod
    def attrs(self) -> Iterator[Attributes]:
        """Iterate over all shard attributes in the dataset in a deterministic order."""
        ...

    @abstractmethod
    def close(self) -> None:
        ...
