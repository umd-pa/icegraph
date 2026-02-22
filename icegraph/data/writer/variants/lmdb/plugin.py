# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import ClassVar, Any, Literal, cast
from pathlib import Path
import msgpack
from datetime import datetime
import lmdb

import pandas as pd

from icegraph import __version__
from icegraph.data.types import Envelope
from icegraph.data.writer import Writer
from icegraph.utils.hashutils import stable_hash_blake2b
from icegraph.types.data import AttributeDomain

from .config import LMDBWriterConfig

# allow msgpack to pack numpy objects
import msgpack_numpy as m
m.patch()

__all__ = ["LMDB"]


_MB = 1 << 20

class LMDB(Writer[LMDBWriterConfig]):
    name: ClassVar[str] = "lmdb"
    version: ClassVar[int] = 1

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> LMDBWriterConfig:
        return LMDBWriterConfig(**config)

    def build(self) -> None:
        return

    def _process(self, env: Envelope) -> None:
        # build output file path
        origin = Path(env.attrs[AttributeDomain.LOCAL.name]["origin"])
        path = self.config.outdir / origin.with_suffix(".lmdb").name

        # add id and set id to attrs
        env.attrs[AttributeDomain.LOCAL.name]["id"] = stable_hash_blake2b(env.main.to_numpy())
        env.attrs[AttributeDomain.GLOBAL.name]["set_id"] = stable_hash_blake2b(env.attrs[AttributeDomain.GLOBAL.name])

        # ensure no stale keys
        if path.exists():
            try:
                path.unlink()
            except OSError as e:
                raise RuntimeError(f"Failed to remove existing LMDB file: {path}") from e

        # get approximate map size requirement
        map_size = self.estimate_map_size(env.main, env.attrs)

        # get handles
        environ, dbs = self.handle(path, map_size)

        try:
            self.write(env, environ, dbs)
        finally:
            environ.close()

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
        return msgpack.packb(value, use_bin_type=True)

    def estimate_map_size(self, main: pd.DataFrame, attrs: dict[str, Any]) -> int:
        # determine size of packed attrs
        # we can ignore ids and info as those are orders of magnitude smaller size and
        # the 3x for headroom will suffice to include them
        size = len(self._pack(attrs))

        # determine size of packed data frame
        sample_count = min(256, len(main))
        if sample_count:
            sample = main.head(sample_count)

            sample_size = 0
            cols = list(sample.columns)
            for row_tuple in sample.itertuples(index=False, name=None):
                # pack the row
                row = dict(zip(cols, row_tuple))

                # record size (4 bytes for uint32be key)
                sample_size += len(self._pack(row)) + 4

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
        chunk_size = 20000
        columns = list(env.main.columns)

        for start in range(0, len(env.main), chunk_size):
            # set next stop checkpoint
            end = min(start + chunk_size, len(env.main))

            with environ.begin(db=dbs["data"], write=True) as txn:
                for i, row_tuple in enumerate(env.main.iloc[start:end].itertuples(index=False, name=None), start=start):
                    # normalize to dict
                    row = dict(zip(columns, row_tuple))

                    # use 4 byte big-endian integer as the key for numeric ordering
                    key = i.to_bytes(4, "big", signed=False)
                    txn.put(key, self._pack(row))

        environ.sync()
