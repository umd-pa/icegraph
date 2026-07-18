# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import ClassVar, Any, Literal, cast
from pathlib import Path
import msgpack
from datetime import datetime
import lmdb

import numpy as np
import polars as pl

from icegraph import __version__
from icegraph.data.envelope import Envelope
from icegraph.data.writer import Writer
from icegraph.utils.hashutils import stable_hash_blake2b
from icegraph.common.data import AttributeDomain

from .config import LMDBWriterConfig

# allow msgpack to pack numpy objects
import msgpack_numpy as m
m.patch()

__all__ = ["LMDB"]

# module logger
import logging
logger = logging.getLogger(__name__)


_MB = 1 << 20

# rows are packed with msgpack-numpy, so nested cells must be materialized as typed numpy arrays
_PL_TO_NP: dict[Any, np.dtype] = {
    pl.Float64: np.dtype(np.float64),
    pl.Float32: np.dtype(np.float32),
    pl.Int64: np.dtype(np.int64),
    pl.Int32: np.dtype(np.int32),
    pl.Int16: np.dtype(np.int16),
    pl.Int8: np.dtype(np.int8),
    pl.UInt64: np.dtype(np.uint64),
    pl.UInt32: np.dtype(np.uint32),
    pl.UInt16: np.dtype(np.uint16),
    pl.UInt8: np.dtype(np.uint8),
    pl.Boolean: np.dtype(np.bool_),
}


class LMDB(Writer[LMDBWriterConfig]):
    name: ClassVar[str] = "lmdb"
    version: ClassVar[int] = 1

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> LMDBWriterConfig:
        return LMDBWriterConfig(**config)

    def build(self) -> None:
        return

    def _process(self, item: Envelope) -> Envelope | None:
        # build output file path
        origin = Path(item.get_local_attr("origin"))
        path = self.config.outdir / origin.with_suffix(".lmdb").name

        # add id and set id to attrs
        _id = stable_hash_blake2b(item.main.rows())
        _set_id = stable_hash_blake2b(item.attrs[AttributeDomain.GLOBAL.name])

        # register ids
        item.set_local_attr("id", _id)
        item.set_global_attr("set_id", _set_id)

        # ensure no stale keys
        if path.exists():
            try:
                path.unlink()
                logger.debug(f"unlinked file at {path!s}")
            except OSError as e:
                raise RuntimeError(f"Failed to remove existing LMDB file: {path}") from e

        # get approximate map size requirement
        map_size = self.estimate_map_size(item.main, item.attrs)

        # get handles
        environ, dbs = self.handle(path, map_size)

        try:
            self.write(item, environ, dbs)
        finally:
            environ.close()

        return item

    @staticmethod
    def handle(path: str | Path, map_size: int) -> tuple[lmdb.Environment, dict[Literal['data', 'attr'], Any]]:
        # initialize the environment
        environ = lmdb.open(
            str(path),
            map_size=map_size,
            lock=True,
            subdir=False,
            max_dbs=2,
            readahead=False
        )

        # open dbs for attrs and data
        dbs = cast(
            dict[Literal['data', 'attr'], Any],
            {key: environ.open_db(key.encode(), create=True) for key in ["data", "attr"]}
        )

        return environ, dbs

    @staticmethod
    def _pack(value: Any) -> bytes:
        # this does return bytes despite warning
        return msgpack.packb(value, use_bin_type=True)  # pyright: ignore[reportReturnType]

    @staticmethod
    def _nested_dtypes(schema: pl.Schema) -> dict[str, np.dtype | None]:
        """Map each nested (List/Array) column to the numpy dtype of its leaf values."""
        nested: dict[str, np.dtype | None] = {}
        for name, dtype in schema.items():
            if not isinstance(dtype, (pl.List, pl.Array)):
                continue

            # walk to the leaf dtype
            leaf = dtype
            while isinstance(leaf, (pl.List, pl.Array)):
                leaf = leaf.inner

            # None leaves conversion to numpy inference
            nested[name] = _PL_TO_NP.get(type(leaf))

        return nested

    @staticmethod
    def _to_record(row: dict[str, Any], nested: dict[str, np.dtype | None]) -> dict[str, Any]:
        """Convert nested cells of a row dict to typed numpy arrays for msgpack-numpy."""
        for name, dtype in nested.items():
            value = row[name]
            if value is not None:
                row[name] = np.asarray(value, dtype=dtype)
        return row

    def estimate_map_size(self, main: pl.DataFrame, attrs: dict[str, Any]) -> int:
        # determine size of packed attrs
        # we can ignore ids and info as those are orders of magnitude smaller size and
        # the 3x for headroom will suffice to include them
        size = len(self._pack(attrs))

        # determine size of packed data frame
        sample_count = min(256, len(main))
        if sample_count:
            nested = self._nested_dtypes(main.schema)

            sample_size = 0
            for row in main.head(sample_count).iter_rows(named=True):
                # pack the row exactly as write() will
                record = self._to_record(row, nested)

                # record size (4 bytes for uint32be key)
                sample_size += len(self._pack(record)) + 4

            average_size = sample_size / sample_count

            # record the average size * row count
            size += int(average_size * len(main))

        size = max(3 * size, 64 * _MB)  # minimum 64 MB, or 3x packed size for headroom
        size = ((size + _MB - 1) // _MB) * _MB  # round up to nearest MB
        return size

    def write(self, env: Envelope, environ: lmdb.Environment, dbs: dict[Literal['data', 'attr'], Any]) -> None:
        # first write attrs
        with environ.begin(db=dbs["attr"], write=True) as txn:
            # write timestamp and icegraph version info
            info = {
                "timestamp": datetime.now().timestamp(),
                "icegraph": {
                    "version": __version__
                },
                "writer": {
                    "name": type(self).name,
                    "version": type(self).version
                },
                "key_encoding": {
                    "data": "uint32be",
                    "attr": "utf-8"
                },
                "packer": "msgpack-numpy",
                "hasher": stable_hash_blake2b.name  # type: ignore
            }
            txn.put("info".encode(), self._pack(info))

            # write env attrs
            for key, attr in env.attrs.items():
                txn.put(key.encode(), self._pack(attr))

        # write data in chunks
        chunk_size = 1000
        nested = self._nested_dtypes(env.main.schema)

        for start in range(0, len(env.main), chunk_size):
            # set next stop checkpoint
            length = min(chunk_size, len(env.main) - start)

            with environ.begin(db=dbs["data"], write=True) as txn:
                for i, row in enumerate(env.main.slice(start, length).iter_rows(named=True), start=start):
                    # normalize nested cells to typed numpy arrays
                    record = self._to_record(row, nested)

                    # use 4 byte big-endian integer as the key for numeric ordering
                    key = i.to_bytes(4, "big", signed=False)
                    txn.put(key, self._pack(record))

        environ.sync()
