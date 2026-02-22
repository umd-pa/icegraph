# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import final, Iterator, Any, ClassVar, TypeVar

from icegraph.types.plugins import Plugin
from icegraph.trainer.services.data.types import Attributes

from .types import ReaderContext

__all__ = ["Reader"]


C = TypeVar("C")


class Reader(Plugin[C, ReaderContext]):
    file_ext: ClassVar[str]

    # satisfy the type checker
    _path: Path
    _attrs: Attributes

    def on_attach(self) -> None:
        self._path = self._ctx.path

        if not self._path.is_file():
            raise FileNotFoundError(f"Path '{self._path!s}' does not resolve to a valid file.")

    @final
    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate through all records."""
        for i in range(len(self)):
            yield self.get(i)

    @final
    def __getitem__(self, index: int | slice) -> dict[str, Any] | list[dict[str, Any]]:
        # handle slices
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            return [self.get(i) for i in range(start, stop, step)]

        # get dataset length
        record_count = len(self)

        # ensure index is an int
        if not isinstance(index, int):
            raise TypeError(f"Index must be int or slice, got {type(index).__name__}.")

        # ensure valid index
        if not (-record_count <= index < record_count):
            raise IndexError(f"Index {index} out of bounds for length {record_count}.")

        # normalize index
        index = index % record_count
        return self.get(index)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __getstate__(self) -> dict[str, Any]:
        # sleep the instance, this closes and removes file handles
        self.sleep()

        # return __dict__
        return self.__dict__.copy()

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)

    @final
    @property
    def attrs(self) -> Attributes:
        if self._attrs is None:
            self._attrs = self._build_attrs()
        return self._attrs

    def close(self) -> None:
        self.sleep()

    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def sleep(self) -> None:
        """Sleep the instance, removing any file handles or pointers."""
        ...

    @abstractmethod
    def get(self, index: int) -> dict[str, Any]:
        ...

    @abstractmethod
    def _build_attrs(self) -> Attributes:
        ...
