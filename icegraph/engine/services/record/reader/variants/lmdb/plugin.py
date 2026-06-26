# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from functools import cached_property
from typing_extensions import Buffer

import lmdb
from typing import Any, ClassVar
import msgpack

from icegraph.common.data import AttributeDomain
from icegraph.common.record import Record, Attributes

from ...reader import Reader

from .config import Config

import msgpack_numpy as m

m.patch()  # allow msgpack to work with numpy objects

__all__ = ["LMDB"]


class LMDB(Reader[Config]):
    name: ClassVar[str] = "lmdb"
    version: ClassVar[int] = 1

    file_ext: ClassVar[str] = ".lmdb"

    def build(self) -> None:
        return

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> Config:
        return Config(**config)

    def record_count(self) -> int:
        with self.env.begin(db=self.dbs["data"]) as txn:
            # this is correct, likely stub issue with lmdb
            return int(txn.stat()["entries"])  # pyright: ignore[reportCallIssue]

    def validate_file(self) -> None:
        with self.env.begin(db=self.dbs["data"]) as txn, txn.cursor() as cur:
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

    @cached_property
    def env(self) -> lmdb.Environment:
        return lmdb.open(
            str(self._path),
            readonly=True,
            lock=False,
            subdir=False,
            max_dbs=2,
            readahead=True
        )

    @cached_property
    def dbs(self) -> dict[str, lmdb._Database]:
        return {key: self.env.open_db(key.encode()) for key in ["data", "attr"]}

    @staticmethod
    def _unpack(value: Buffer, **kwargs) -> Any:
        """Unpack a byte value using msgpack."""
        return msgpack.unpackb(value, object_hook=m.decode, raw=False, **kwargs)

    def _build_attrs(self) -> Attributes:
        """Build an Attributes object."""
        attrs: dict[AttributeDomain, Any] = {}
        with self.env.begin(db=self.dbs["attr"]) as txn:
            for domain in AttributeDomain.all():
                # load attrs from file
                attrs_bytes = txn.get(domain.name.encode())

                if attrs_bytes is None:
                    # key not found, raise
                    raise KeyError(f"Key '{domain.name}' not found in LMDB file {self._path}.")

                attrs[domain] = self._unpack(attrs_bytes)

        return Attributes(attrs)

    def get(self, index: int) -> Record:
        # pack to unsigned 4 byte big-endian
        key = index.to_bytes(4, "big", signed=False)

        with self.env.begin(db=self.dbs["data"]) as txn:
            data = txn.get(key)

        # ensure data exists
        if data is None:
            raise KeyError(f"Index {index} not found in file {self._path}")

        # build sample
        record = Record(
            index=index,
            shard_id=self.attrs.shard_id,
            data=self._unpack(data)
        )

        return record

    def sleep(self) -> None:
        # close the environment and remove all references
        vars(self).pop("dbs", None)
        env = vars(self).pop("env", None)
        if env is not None:
            env.close()
