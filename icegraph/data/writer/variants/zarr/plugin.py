# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import ClassVar, Any
from pathlib import Path
import math

import zarr
import numpy as np
import polars as pl

from icegraph.data.writer import Writer

from .config import ZarrWriterConfig

__all__ = ["Zarr"]

# module logger
import logging
logger = logging.getLogger(__name__)


class Zarr(Writer[ZarrWriterConfig]):
    name: ClassVar[str] = "zarr"
    version: ClassVar[int] = 1

    # writer specific vars
    suffix: ClassVar[str] = ".zarr"

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> ZarrWriterConfig:
        return ZarrWriterConfig(**config)

    def build(self) -> None:
        return

    def _write_data(self, root: zarr.Group, data: pl.DataFrame) -> None:
        values = root.require_group("values")
        offsets = root.require_group("offsets")

        for name, dtype in data.schema.items():
            col = data.get_column(name)

            # complex case, ragged list with non arange offsets
            if isinstance(dtype, pl.List):
                # get size of each item in list
                lengths = col.list.len().to_numpy().astype(np.int64)

                # allocate offsets array
                off = np.zeros(data.height + 1, np.int64)

                # populate with cumsum over lengths
                np.cumsum(lengths, out=off[1:])

                # flatten list to array
                arr = col.explode().to_numpy()

                # ensure sizes match expected
                if arr.shape[0] != off[-1]:
                    raise ValueError(
                        f"{type(self).__name__}: {name}: explode produced {arr.shape[0]} rows, offsets expect {off[-1]}. "
                        "Column likely contains empty or null lists."
                    )

            # simple case, offsets are just arange
            else:
                off = np.arange(data.height + 1, dtype=np.int64)
                arr = col.to_numpy()

            if arr.dtype == object and dtype != pl.String:
                raise ValueError(f"{name}: {dtype} produced object dtype after flattening")

            # target bytes per chunk, rows derived from the row width
            row_bytes = arr.dtype.itemsize * int(np.prod(arr.shape[1:], dtype=np.int64))
            rows = max(1, min(arr.shape[0], (self.config.chunk_size * 1024 ** 2) // row_bytes))

            # one shard per array, rounded up to a whole number of chunks
            shard_rows = max(rows, math.ceil(arr.shape[0] / rows) * rows)

            # write values as array
            values.create_array(
                name,
                shape=arr.shape,
                dtype=str if arr.dtype == object else arr.dtype,
                chunks=(rows,) + arr.shape[1:],
                shards=(shard_rows,) + arr.shape[1:],
            )[...] = arr

            # write offsets as array
            off_chunk = min(len(off), 65536)
            offsets.create_array(
                name,
                shape=off.shape,
                dtype=off.dtype,
                chunks=(off_chunk,),
                shards=(max(off_chunk, math.ceil(len(off) / off_chunk) * off_chunk),),
            )[...] = off

    def _write_schema(self, root: zarr.Group, data: pl.DataFrame) -> None:
        blob = np.frombuffer(data.clear().serialize(), dtype=np.uint8)
        root.require_array("_schema", shape=blob.shape, dtype=blob.dtype)[...] = blob

    def _write(self, table: pl.DataFrame, metadata: dict[str, Any], fp: Path) -> None:
        root = zarr.open_group(fp, mode="w", attributes=metadata)

        # write schema
        self._write_schema(root, table)

        # write data
        self._write_data(root, table)

