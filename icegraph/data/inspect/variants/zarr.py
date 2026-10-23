# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Any
from functools import cached_property
import io

import zarr
import numpy as np
import polars as pl
import pyarrow as pa

from icegraph.common.data import restore

from ..inspector import Inspector

__all__ = ["ZarrInspector"]


class ZarrInspector(Inspector):

    def build(self) -> None:
        pass

    @cached_property
    def _root(self) -> zarr.Group:
        return zarr.open_group(self._path, mode="r")

    @cached_property
    def _values(self) -> zarr.Group:
        return self._root.require_group("values")

    @cached_property
    def _offsets(self) -> zarr.Group:
        return self._root.require_group("offsets")

    @staticmethod
    def _read(group: zarr.Group, name: str) -> np.ndarray:
        """Read a member as a numpy array."""
        obj = group[name]
        if not isinstance(obj, zarr.Array):
            raise TypeError(f"expected an array at {name!r}, found a group")
        return np.asarray(obj[:])

    @cached_property
    def _schema(self) -> pl.Schema:
        raw = self._read(self._root, "_schema").tobytes()
        return pl.DataFrame.deserialize(io.BytesIO(raw)).schema

    @staticmethod
    def _rebuild(name: str, values: np.ndarray, offsets: np.ndarray, nested: bool) -> pl.Series:
        """Reconstruct one column from its flat values and offsets."""
        # not a list column, so the values are the column
        if not nested:
            return pl.Series(name, values)

        # arrow stores a list column as exactly this: a child array plus offsets
        flat = pa.array(values.reshape(-1))
        child = flat if values.ndim == 1 else pa.FixedSizeListArray.from_arrays(flat, values.shape[1])

        return pl.Series(name, pl.from_arrow(pa.ListArray.from_arrays(pa.array(offsets), child)))

    def _load_df(self) -> pl.DataFrame:
        return pl.DataFrame({
            name: self._rebuild(
                name,
                self._read(self._values, name),
                self._read(self._offsets, name),
                isinstance(dtype, pl.List),
            )
            for name, dtype in self._schema.items()
        })

    def _load_attrs(self) -> dict[str, Any]:
        return restore(dict(self._root.attrs))