# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from pathlib import Path

from dataclasses import dataclass
from functools import cached_property
from typing_extensions import Buffer

import lmdb
from typing import Any, ClassVar
import msgpack
import numpy as np

from icegraph.common.record import RecordBlock, Column
from icegraph.typing.common import ArrayI
from icegraph.common.data import restore

from ...reader import Reader

from .config import Config

import msgpack_numpy as m

m.patch()  # allow msgpack to work with numpy objects

__all__ = ["LMDB"]


@dataclass(frozen=True)
class Handle:
    env: lmdb.Environment
    dbs: dict[str, lmdb._Database]

    def close(self) -> None:
        vars(self).pop("dbs", None)
        env = vars(self).pop("env", None)
        if env is not None:
            env.close()


class LMDB(Reader[Config, Handle]):
    name: ClassVar[str] = "lmdb"
    version: ClassVar[int] = 1

    file_ext: ClassVar[str] = ".lmdb"

    def build(self) -> None:
        return

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> Config:
        return Config(**config)

    def validate_file(self) -> None:
        with self.handle.env.begin(db=self.handle.dbs["data"]) as txn, txn.cursor() as cur:
            if not cur.first():
                raise KeyError("Database is empty, cannot find first key.")
            first = int.from_bytes(cur.key(), "big", signed=False)

            if not cur.last():
                raise KeyError("Database is empty, cannot find last key.")
            last = int.from_bytes(cur.key(), "big", signed=False)

        if not (first == 0 and last == len(self) - 1):
            raise RuntimeError(
                "Database failed first/last sanity check. All records must be contiguous, indexed from 0 to N-1."
            )

    def _open(self, path: Path) -> Handle:
        # open file handle
        env = lmdb.open(
            str(path),
            readonly=True,
            lock=False,
            subdir=False,
            max_dbs=2,
            readahead=True
        )

        # open database specific handles
        dbs = {key: env.open_db(key.encode()) for key in ["data", "attr"]}

        return Handle(env, dbs)

    def _close(self, handle: Handle) -> None:
        handle.close()

    @staticmethod
    def _unpack(value: Buffer, **kwargs) -> Any:
        """Unpack a byte value using msgpack."""
        return msgpack.unpackb(value, object_hook=m.decode, raw=False, **kwargs)

    @cached_property
    def _attrs_dict(self) -> dict[str, Any]:
        """Build an Attributes object."""
        attrs: dict[str, Any] = {}
        with self.handle.env.begin(db=self.handle.dbs["attr"]) as txn:
            for key, value in txn.cursor():
                attrs[key.decode()] = self._unpack(value)

        return attrs

    def _get(self, indices: ArrayI) -> RecordBlock:
        rows: list[dict[str, Any]] = []
        with self.handle.env.begin(db=self.handle.dbs["data"]) as txn:
            for index in indices:
                # np ints lack .to_bytes, coerce to plain int
                key = int(index).to_bytes(4, "big", signed=False)
                data = txn.get(key)

                if data is None:
                    raise KeyError(f"Index {index} not found in file {self._ctx.path}")

                rows.append(self._unpack(data))

        return self._to_block(rows)

    @staticmethod
    def _to_block(rows: list[dict[str, Any]]) -> RecordBlock:
        """Stack row-oriented records into a columnar block."""
        height = len(rows)

        columns: dict[str, Column] = {}
        for name in rows[0] if rows else ():
            values = [np.asarray(row[name]) for row in rows]

            lengths = np.fromiter((v.shape[0] for v in values), dtype=np.int64)

            offsets = np.zeros(height + 1, np.int64)
            np.cumsum(lengths, out=offsets[1:])

            columns[name] = Column(np.concatenate(values, axis=0), offsets)

        return RecordBlock(height=height, columns=columns)
