# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

import lmdb
from typing import Any, ClassVar
import msgpack

from icegraph.types.data import AttributeDomain
from icegraph.trainer.services.data.types import Attributes

from ...reader import Reader

from .config import Config

import msgpack_numpy as m

m.patch()  # allow msgpack to work with numpy objects

__all__ = ["LMDB"]


class LMDB(Reader[Config]):
    name: ClassVar[str] = "lmdb"
    version: ClassVar[int] = 1

    file_ext: ClassVar[str] = ".lmdb"

    _env: lmdb.Environment | None
    _dbs: dict[str, Any] | None

    def build(self) -> None:
        # init the reader env and db handles cache
        self._env = None
        self._dbs = None

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> Config:
        return Config(**config)

    def _get_len(self) -> int:
        with self.env.begin(db=self.dbs["data"]) as txn:
            return txn.stat()["entries"]

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

        # ensure handles are closed after completion
        self.sleep()

    @property
    def env(self) -> lmdb.Environment:
        if self._env is None:
            self._env = lmdb.open(
                str(self._path),
                readonly=True,
                lock=False,
                subdir=False,
                max_dbs=2,
                readahead=True
            )
        return self._env

    @property
    def dbs(self) -> dict[str, Any]:
        if self._dbs is None:
            self._dbs = {
                key: self.env.open_db(key.encode()) for key in ["data", "attr"]
            }
        return self._dbs

    @staticmethod
    def _unpack(value: bytes, **kwargs) -> Any:
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

        # need to sleep so we dont leave handles open
        self.sleep()

        return Attributes(attrs)

    def get(self, index: int) -> dict[str, Any]:
        # pack to unsigned 4 byte big-endian
        key = index.to_bytes(4, "big", signed=False)

        with self.env.begin(db=self.dbs["data"]) as txn:
            data = txn.get(key)

        # ensure data exists
        if data is None:
            raise KeyError(f"Index {index} not found in file {self._path}")

        return self._unpack(data)

    def sleep(self) -> None:
        # close the environment
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass

        # remove all references
        self._dbs = None
        self._env = None

        # close cached attrs so they don't blow up memory
        self._attrs = None
