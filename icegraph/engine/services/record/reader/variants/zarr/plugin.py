# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from pathlib import Path

from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, ClassVar
from collections.abc import Collection

import zarr
import io
import polars as pl
import numpy as np

from icegraph.common.record import RecordBlock, Column
from icegraph.typing.common import ArrayI
from icegraph.common.data import restore

from ...reader import Reader

from .config import Config

__all__ = ["Zarr"]


@dataclass(frozen=True)
class Handle:
    root: zarr.Group

    # a zarr group re-fetches and re-parses zarr.json on every member lookup, so
    # resolving an array per read costs two store round-trips before any data
    # moves, the hot path resolves each array once and keeps it
    _values: dict[str, zarr.Array] = field(default_factory=dict, repr=False, compare=False)

    def value_array(self, name: str) -> zarr.Array:
        """Resolve a values array once, then serve it from the handle."""
        array = self._values.get(name)

        if array is None:
            array = self._values[name] = self.get_array(name, self.values)

        return array

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
        return {
            name: np.asarray(self.handle.get_array(name, self.handle.offsets)[:])
            for name in self._schema
        }

    @cached_property
    def _attrs_dict(self) -> dict[str, Any]:
        """Build an Attributes object."""
        return restore(dict(self.handle.root.attrs))

    def _read_rows(self, name: str, starts: ArrayI, stops: ArrayI) -> np.ndarray:
        """Read the row segments ``[starts[i], stops[i])``, concatenated."""
        array = self.handle.value_array(name)

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

        if total >= (hi - lo) * self.config.dense_read_fraction:
            return np.asarray(array[lo:hi])[rows - lo]

        return np.asarray(array.oindex[rows])

    def _selected(self, columns: Collection[str] | None) -> list[str]:
        """Requested columns that the file actually holds, in schema order."""
        if columns is None:
            return list(self._schema)

        # an absent column is not an error: the decode service already treats a
        # missing column as an empty role, so intersect rather than raise
        return [name for name in self._schema if name in columns]

    def _get(self, indices: ArrayI, columns: Collection[str] | None = None) -> RecordBlock:
        block: dict[str, Column] = {}

        for name in self._selected(columns):
            off = self._offsets[name]
            starts, stops = off[indices], off[indices + 1]

            values = self._read_rows(name, starts, stops)

            # offsets local to this block
            local = np.zeros(len(indices) + 1, np.int64)
            np.cumsum(stops - starts, out=local[1:])

            block[name] = Column(values, local)

        return RecordBlock(height=len(indices), columns=block)
