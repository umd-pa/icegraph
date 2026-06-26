# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import final, Iterator, Any, ClassVar, TypeVar, overload
from functools import cached_property

import numpy as np

from icegraph.common.plugins import Plugin
from icegraph.common.record import Attributes, Record

from .types import ReaderContext

__all__ = ["Reader"]


C = TypeVar("C")


class Reader(Plugin[C, ReaderContext]):
    file_ext: ClassVar[str]

    def on_attach(self) -> None:
        if not self._path.is_file():
            raise FileNotFoundError(f"Path '{self._path!s}' does not resolve to a valid file.")

        self.validate_file()
        self.sleep()

    @final
    def __len__(self) -> int:
        # build using subclass logic
        count = self.record_count()

        # make sure handles are closed
        self.sleep()

        return count

    @final
    def __iter__(self) -> Iterator[Record]:
        """Iterate through all records."""
        for i in range(len(self)):
            yield self.get(i)

    @overload
    def __getitem__(self, index: int) -> Record:...

    @overload
    def __getitem__(self, index: slice) -> list[Record]:...

    @final
    def __getitem__(self, index: int | slice) -> Record | list[Record]:
        # handle slices
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            return [self.get(i) for i in range(start, stop, step)]

        # get dataset length
        record_count = len(self)

        # ensure index is an int
        if not isinstance(index, (int, np.integer)):
            raise TypeError(f"Index must be int or slice, got {type(index).__name__}.")

        # ensure valid index
        if not (-record_count <= index < record_count):
            raise IndexError(f"Index {index} out of bounds for length {record_count}.")

        # normalize index
        index = int(index % record_count)
        return self.get(index)

    def __del__(self) -> None:
        self.close()

    def __getstate__(self) -> dict[str, Any]:
        # sleep the instance, this closes and removes file handles
        self.sleep()

        # return __dict__
        return self.__dict__.copy()

    def __setstate__(self, state: dict[str, Any]) -> None:
        vars(self).update(state)

    def validate_file(self) -> None:
        pass

    @cached_property
    def _path(self) -> Path:
        return self._ctx.path

    @final
    @cached_property
    def attrs(self) -> Attributes:
        # build using subclass logic
        attrs = self._build_attrs()

        # make sure handles are closed
        self.sleep()

        return attrs

    def close(self) -> None:
        self.sleep()

    @abstractmethod
    def record_count(self) -> int:
        ...

    @abstractmethod
    def sleep(self) -> None:
        """Sleep the instance, removing any file handles or pointers."""
        ...

    @abstractmethod
    def get(self, index: int) -> Record:
        ...

    @abstractmethod
    def _build_attrs(self) -> Attributes:
        ...
