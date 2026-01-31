# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

import lmdb
from typing import Any
from pathlib import Path
import msgpack
import struct

from icegraph.types.data import AttributeDomain
from icegraph.types.common import ArrayG
from icegraph.trainer.services.data.types import Attributes

from ..reader import Reader

import msgpack_numpy as m
m.patch()  # allow msgpack to work with numpy objects

__all__ = ["LMDB"]


class LMDB(Reader):
    name: str = "lmdb"
    file_ext: str = "lmdb"

    def __init__(self, path: str | Path) -> None:
        super().__init__(path)

        # init the reader env and db handles cache
        self._env: lmdb.Environment | None = None
        self._dbs: dict[str, Any]   | None = None

        # assert keys are contiguous (very lightweight check)
        self._assert_contiguous()

    def _assert_contiguous(self) -> None:
        with self.env.begin(db=self.dbs["data"]) as txn, txn.cursor() as cur:
            if not cur.first():
                raise KeyError("Database is empty, cannot find first key.")
            first = struct.unpack(">Q", cur.key())[0]

            if not cur.last():
                raise KeyError("Database is empty, cannot find last key.")
            last = struct.unpack(">Q", cur.key())[0]

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
                str(self.path),
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

    def __len__(self) -> int:
        with self.env.begin(db=self.dbs["data"]) as txn:
            return txn.stat()["entries"]

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
                attrs_bytes = txn.get(domain.value.encode())

                if attrs_bytes is None:
                    # key not found, raise
                    raise KeyError(f"Key '{domain.value}' not found in LMDB file {self.path}.")

                attrs[domain] = self._unpack(attrs_bytes)

        return Attributes(attrs)

    def get(self, index: int) -> dict[str, ArrayG]:
        # pack to big-endian
        key = struct.pack(">Q", index)

        with self.env.begin(db=self.dbs["data"]) as txn:
            data = txn.get(key)

        # ensure data exists
        if data is None:
            raise KeyError(f"Index {index} not found in file {self.path}")

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

        # close caches attrs so they don't blow up memory
        self._attrs = None
