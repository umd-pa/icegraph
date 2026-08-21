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

from icegraph.common.record import Record
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
        return {
            name: np.asarray(self.handle.get_array(name, self.handle.offsets)[:])
            for name in self._schema
        }

    @cached_property
    def _attrs_dict(self) -> dict[str, Any]:
        """Build an Attributes object."""
        return restore(dict(self.handle.root.attrs))

    def _get(self, indices: ArrayI) -> list[Record]:
        # ascending for the read, remember where each came from
        order = np.argsort(indices, kind="stable")
        ordered = indices[order]

        columns: dict[str, tuple[np.ndarray, np.ndarray, bool]] = {}

        for name, dtype in self._schema.items():
            # mirrors the writer: only List columns get non-arange offsets
            ragged = isinstance(dtype, pl.List)

            off = self._offsets[name]
            starts, stops = off[ordered], off[ordered + 1]

            # read only the rows this batch needs
            rows = np.concatenate([np.arange(s, e) for s, e in zip(starts, stops)])
            flat = np.asarray(self.handle.get_array(name, self.handle.values).oindex[rows])

            # offsets local to this batch
            local = np.zeros(len(ordered) + 1, np.int64)
            np.cumsum(stops - starts, out=local[1:])

            columns[name] = (flat, local, ragged)

        shard_id = self.attrs.shard_id

        records = [
            Record(
                index=int(index),
                shard_id=shard_id,
                data={
                    name: flat[local[i]:local[i + 1]] if ragged else flat[local[i]]
                    for name, (flat, local, ragged) in columns.items()
                },
            )
            for i, index in enumerate(ordered)
        ]

        # invert the permutation to restore requested order
        inverse = np.empty_like(order)
        inverse[order] = np.arange(order.size)

        return [records[i] for i in inverse]
