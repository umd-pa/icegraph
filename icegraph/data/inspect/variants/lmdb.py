# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Any

import lmdb
import pandas as pd
import msgpack

from ..inspector import Inspector

# so we can load np arrays
import msgpack_numpy as m
m.patch()

__all__ = ["LMDBInspector"]


class LMDBInspector(Inspector):

    _env: lmdb.Environment
    _dbs: dict[str, Any]

    def build(self) -> None:
        # build env and dbs once, no need for lazy loading
        self._env = lmdb.open(
            str(self._path),
            readonly=True,
            lock=False,
            subdir=False,
            max_dbs=2,
            readahead=True
        )

        self._dbs = {
            key: self._env.open_db(key.encode()) for key in ["data", "attr"]
        }

    @staticmethod
    def _unpack(value: bytes, **kwargs) -> Any:
        """Unpack a byte value using msgpack."""
        return msgpack.unpackb(value, object_hook=m.decode, raw=False, **kwargs)

    def _load_df(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        with self._env.begin(db=self._dbs["data"]) as txn, txn.cursor() as cur:
            # keys are int index, so dont need to unpack them here
            for _, v in cur:
                rows.append(self._unpack(v))

        return pd.DataFrame(rows)

    def _load_attrs(self) -> dict[str, Any]:
        attrs: dict[str, Any] = {}
        with self._env.begin(db=self._dbs["attr"]) as txn, txn.cursor() as cur:
            for k, v in cur:
                attrs[k.decode()] = self._unpack(v)

        return attrs

    def close(self) -> None:
        self._env.close()
