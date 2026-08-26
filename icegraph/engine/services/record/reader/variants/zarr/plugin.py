# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from pathlib import Path

from dataclasses import dataclass
from functools import cached_property
from typing import Any, ClassVar

import zarr
import io
import polars as pl
import numpy as np

from icegraph.common.record import Column, RecordBlock
from icegraph.typing.common import ArrayI
from icegraph.common.data import restore

from ...reader import Reader

from .config import Config

__all__ = ["Zarr"]


@dataclass(frozen=True)
class Handle:
    root: zarr.Group

    # sub dbs
    @property
    def values(self) -> zarr.Group:
        return self.get_group("values", self.root)

    @property
    def offsets(self) -> zarr.Group:
        return self.get_group("offsets", self.root)

    @staticmethod
    def get_array(name: str, from_group: zarr.Group) -> zarr.Array:
        obj = from_group[name]

        if not isinstance(obj, zarr.Array):
            raise TypeError(f"Expected an array at {name!r}, found {type(obj)}.")

        return obj

    @staticmethod
    def get_group(name: str, from_group: zarr.Group) -> zarr.Group:
        obj = from_group[name]

        if not isinstance(obj, zarr.Group):
            raise TypeError(f"Expected a group at {name!r}, found {type(obj)}.")

        return obj

    def close(self) -> None:
        self.root.store.close()


class Zarr(Reader[Config, Handle]):
    name: ClassVar[str] = "zarr"
    version: ClassVar[int] = 1

    file_ext: ClassVar[str] = ".zarr"

    def build(self) -> None:
        return

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> Config:
        return Config(**config)

    def _open(self, path: Path) -> Handle:
        return Handle(zarr.open_group(path, mode="r"))

    def _close(self, handle: Handle) -> None:
        vars(self).pop("_offsets", None)
        handle.close()

    @cached_property
    def _schema(self) -> pl.Schema:
        raw = np.asarray(
            self.handle.get_array("_schema", self.handle.root)[:]
        ).tobytes()
        return pl.DataFrame.deserialize(io.BytesIO(raw)).schema

    @cached_property
    def _offsets(self) -> dict[str, ArrayI]:
        # only List columns have non-arange offsets; plain columns are indexed directly
        return {
            name: np.asarray(self.handle.get_array(name, self.handle.offsets)[:])
            for name, dtype in self._schema.items()
            if isinstance(dtype, pl.List)
        }

    @cached_property
    def _attrs_dict(self) -> dict[str, Any]:
        """Build an Attributes object."""
        return restore(dict(self.handle.root.attrs))

    # covering-slice reads are one sequential request; fall back to exact row
    # gathers only when the needed rows are a small fraction of the span
    _DENSE_READ_FRACTION: ClassVar[float] = 0.5

    def _read_rows(self, name: str, starts: ArrayI, stops: ArrayI) -> np.ndarray:
        """Read the row segments ``[starts[i], stops[i])``, concatenated."""
        array = self.handle.get_array(name, self.handle.values)

        lengths = stops - starts
        local = np.zeros(len(starts) + 1, np.int64)
        np.cumsum(lengths, out=local[1:])
        total = int(local[-1])

        if total == 0:
            return np.empty((0,) + array.shape[1:], dtype=array.dtype)

        lo, hi = int(starts[0]), int(stops[-1])

        # segments are disjoint and ascending, so total == span means no gaps
        if total == hi - lo:
            return np.asarray(array[lo:hi])

        # positions of the needed rows, relative to the covering span
        rows = np.arange(total, dtype=np.int64) + np.repeat(starts - local[:-1], lengths)

        if total >= (hi - lo) * self._DENSE_READ_FRACTION:
            return np.asarray(array[lo:hi])[rows - lo]

        return np.asarray(array.oindex[rows])

    def _get(self, indices: ArrayI) -> RecordBlock:
        columns: dict[str, Column] = {}

        for name, dtype in self._schema.items():
            # mirrors the writer: only List columns get non-arange offsets
            if isinstance(dtype, pl.List):
                off = self._offsets[name]
                starts, stops = off[indices], off[indices + 1]

                values = self._read_rows(name, starts, stops)

                # offsets local to this block
                local = np.zeros(len(indices) + 1, np.int64)
                np.cumsum(stops - starts, out=local[1:])

                columns[name] = Column(values, local)
            else:
                values = self._read_rows(name, indices, indices + 1)
                columns[name] = Column(values)

        return RecordBlock(height=len(indices), columns=columns)
