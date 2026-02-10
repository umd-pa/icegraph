# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from abc import ABC, abstractmethod
from pathlib import Path
from typing import final, Iterator, Any, ClassVar

from icegraph.types.common import ArrayG

from ..types import Attributes

__all__ = ["Reader"]


class Reader(ABC):
    name: ClassVar[str]
    file_ext: ClassVar[str]

    def __init__(self, path: str | Path) -> None:
        # cache the path
        self.path = Path(path)

        # verify the path actually points to a file
        if not self.path.is_file():
            raise FileNotFoundError(f"Path '{self.path!s}' does not resolve to a valid file.")

        # cache for shard attributes
        self._attrs: Attributes | None = None

        # cache for shard id
        self._shard_id: str | None = None

    @abstractmethod
    def __len__(self) -> int:
        ...

    @final
    def __iter__(self) -> Iterator[dict[str, ArrayG]]:
        """Iterate through all records."""
        for i in range(len(self)):
            yield self.get(i)

    @final
    def __getitem__(self, index: int | slice) -> dict[str, ArrayG] | list[dict[str, ArrayG]]:
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

    @abstractmethod
    def get(self, index: int) -> dict[str, ArrayG]:
        ...

    @abstractmethod
    def _build_attrs(self) -> Attributes:
        ...

    @final
    @property
    def attrs(self) -> Attributes:
        if self._attrs is None:
            self._attrs = self._build_attrs()
        return self._attrs

    @abstractmethod
    def sleep(self) -> None:
        """Sleep the instance, removing any file handles or pointers."""
        ...

    def close(self) -> None:
        self.sleep()
