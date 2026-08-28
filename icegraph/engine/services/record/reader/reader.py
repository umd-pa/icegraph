# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import Generic, final, Any, ClassVar, TypeVar
from functools import cached_property

import numpy as np

from icegraph.common.plugins import Plugin
from icegraph.common.record import Attributes, RecordBlock
from icegraph.typing.common import ArrayI
from icegraph.common.data import AttributeDomain

from .types import ReaderContext

import logging
logger = logging.getLogger(__name__)

__all__ = ["Reader"]


C = TypeVar("C")
_HANDLE = TypeVar("_HANDLE")


class Reader(Plugin[C, ReaderContext], Generic[C, _HANDLE]):
    file_ext: ClassVar[str]

    def on_attach(self) -> None:
        if not self._ctx.path.exists():
            raise FileNotFoundError(f"Path '{self._ctx.path!s}' does not resolve to a valid file or directory.")

        self.validate_file()
        self.close()

    @final
    def __len__(self) -> int:
        return self.sample_count

    @final
    def __getitem__(self, indices: ArrayI) -> RecordBlock:
        """Read the given rows as one columnar block.

        Indices must be ascending; callers (the record service) sort once so
        readers can turn each request into large sequential reads.
        """
        # ensure indices is array
        if not isinstance(indices, np.ndarray):
            raise TypeError(f"Indices must be an npt.NDArray, got {type(indices).__name__}.")

        # ensure array contains ints
        if not np.issubdtype(indices.dtype, np.integer):
            raise TypeError(f"Indices must be an npt.NDArray[np.integer], got {indices.dtype}")

        # ensure array is 1 dim
        if indices.ndim > 1:
            raise ValueError(f"Indices must be an array with exactly 1 dim, got {indices.ndim}")

        # ensure valid index
        out_of_bounds = (indices < -len(self)) | (indices >= len(self))

        if out_of_bounds.any():
            raise IndexError(f"Index {indices[out_of_bounds]} out of bounds for dataset of length {len(self)}.")

        # normalize index
        return self._get(indices)

    def __getstate__(self) -> dict[str, Any]:
        # close before pickle
        self.close()

        return self.__dict__.copy()

    def __setstate__(self, state: dict[str, Any]) -> None:
        vars(self).update(state)

    @cached_property
    def sample_count(self) -> int:
        count = self._attrs_dict["entries"]

        if isinstance(count, np.ndarray):
            count = int(count)

        if not isinstance(count, int):
            raise TypeError(f"root.attrs.entries must be an int, got {count} ({type(count)}).")

        if count < 0:
            raise TypeError(f"root.attrs.entries must be a positive int, got {count} ({type(count)}).")

        if count == 0:
            raise RuntimeError("Received a file with no entries.")

        return count

    @cached_property
    def handle(self) -> _HANDLE:
        return self._open(self._ctx.path)

    def validate_file(self) -> None:
        pass

    @cached_property
    def attrs(self) -> Attributes:
        data = self._attrs_dict

        # normalize keys to AttributeDomain
        attrs = {
            AttributeDomain.LOCAL: data["LOCAL"],
            AttributeDomain.GLOBAL: data["GLOBAL"]
        }

        return Attributes(attrs)

    @cached_property
    @abstractmethod
    def _attrs_dict(self) -> dict[str, Any]:
        ...

    @abstractmethod
    def _open(self, path: Path) -> _HANDLE:
        ...

    @abstractmethod
    def _close(self, handle: _HANDLE) -> None:
        ...

    @abstractmethod
    def _get(self, indices: ArrayI) -> RecordBlock:
        ...

    def close(self) -> None:
        if "handle" not in vars(self):
            return

        self._close(self.handle)
        vars(self).pop("handle", None)
