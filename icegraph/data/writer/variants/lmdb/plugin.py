# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import ClassVar, Any, Literal, cast, Iterator
from pathlib import Path
import msgpack
from datetime import datetime
import lmdb

import numpy as np
import polars as pl

from icegraph import __version__
from icegraph.data.envelope import Envelope
from icegraph.data.writer import Writer
from icegraph.utils.hashutils import CBORBlake2B
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


class LMDB(Writer[LMDBWriterConfig]):
    name: ClassVar[str] = "lmdb"
    version: ClassVar[int] = 1
    suffix = ".lmdb"

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> LMDBWriterConfig:
        return LMDBWriterConfig(**config)

    def build(self) -> None:
        return

    def _process(self, item: Envelope) -> Envelope | None:
        # build output file path
        origin = Path(item.get_local_attr("origin"))
        path = self.outdir / origin.with_suffix(".lmdb").name

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
            self.write(item, environ, dbs, hasher="name")
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
    def _records(main: pl.DataFrame) -> Iterator[dict[str, Any]]:
        """Yield per-row dicts."""
        columns = main.columns
        schema = main.schema

        nested_cols = [n for n, dt in schema.items() if isinstance(dt, (pl.List, pl.Array))]
        scalar_cols = [n for n in columns if n not in nested_cols]

        # one dtype-correct numpy array per nested column
        # an object array of typed arrays for List columns, or a single typed ND array for
        # fixed-width Array columns
        nested = {n: main.get_column(n).to_numpy() for n in nested_cols}

        # scalars are iterated natively, nested cells are pulled from nested,
        # so the large graph payloads are materialized only once
        scalar_iter = main.select(scalar_cols).iter_rows(named=True) if scalar_cols else None

        for i in range(len(main)):
            scal = next(scalar_iter) if scalar_iter is not None else {}

            row: dict[str, Any] = {}
            for name in columns:
                row[name] = nested[name][i] if name in nested else scal[name]

            yield row

    def estimate_map_size(self, main: pl.DataFrame, attrs: dict[str, Any]) -> int:
        # determine size of packed attrs
        # we can ignore ids and info as those are orders of magnitude smaller size and
        # the 3x for headroom will suffice to include them
        size = len(self._pack(attrs))

        # determine size of packed data frame
        sample_count = min(256, len(main))
        if sample_count:
            sample_size = 0
            for record in self._records(main.head(sample_count)):
                # pack the row exactly as write() will (4 bytes for uint32be key)
                sample_size += len(self._pack(record)) + 4

            average_size = sample_size / sample_count

            # record the average size * row count
            size += int(average_size * len(main))

        size = max(3 * size, 64 * _MB)  # minimum 64 MB, or 3x packed size for headroom
        size = ((size + _MB - 1) // _MB) * _MB  # round up to nearest MB
        return size

    def write(self, env: Envelope, environ: lmdb.Environment, dbs: dict[Literal['data', 'attr'], Any], hasher: str) -> None:

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
                "hasher": hasher
            }
            txn.put("info".encode(), self._pack(info))

            # write env attrs
            for key, attr in env.attrs.items():
                txn.put(key.encode(), self._pack(attr))

        # write data in chunks
        chunk_size = 1000

        for start in range(0, len(env.main), chunk_size):
            # set next stop checkpoint
            length = min(chunk_size, len(env.main) - start)

            with environ.begin(db=dbs["data"], write=True) as txn:
                for offset, record in enumerate(self._records(env.main.slice(start, length))):
                    # use 4 byte big-endian integer as the key for numeric ordering
                    key = (start + offset).to_bytes(4, "big", signed=False)
                    txn.put(key, self._pack(record))

        environ.sync()
