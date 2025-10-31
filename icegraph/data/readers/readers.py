# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from dataclasses import dataclass
import struct
import lmdb
from pathlib import Path
from typing import List, Tuple, Union, Sequence, Iterator, Dict, Optional, overload, Self, Any, ClassVar
from collections import OrderedDict
import msgpack
from bisect import bisect_right

import pandas as pd
import numpy as np

from icegraph.utils.pathutils import PathResolver, PathValidator

import msgpack_numpy as m
m.patch()  # allow msgpack to work with numpy objects

__all__ = ["LMDBDatasetShardReader", "LMDBReader"]


class LMDBDatasetShardReader:
    """
    Provides access to one or more LMDB files from the same dataset written using 8-byte big endian integer keys
    """

    @dataclass
    class _Handle:
        env: lmdb.Environment
        dtxn: lmdb.Transaction
        atxn: lmdb.Transaction

        def __del__(self):
            self.close()

        def close(self):
            for txn in (self.atxn, self.dtxn):
                try:
                    txn.abort()
                except Exception:
                    pass
            try:
                self.env.close()
            except Exception:
                pass

    def __init__(self, source: Union[str, Path, Sequence[Union[str, Path]]], *, max_open_envs: int = 256) -> None:
        """
        Initialize the shard reader.
        """
        self._lmdb_paths:       Tuple[Path, ...]        = tuple(PathResolver.normalize_sources(source, ".lmdb"))
        self._index_arr:        Optional[np.ndarray]    = None  # lazy build on first __getitem__ call
        self._max_open_envs:    int                     = max_open_envs

        # cache for open env/txn
        self._cache: OrderedDict[Path, LMDBDatasetShardReader._Handle] = OrderedDict()

        # attributes cache
        self._attributes: Optional[Dict[bytes, Dict[str, Dict[str, Any]]]] = None

        # index data type
        self._ID_BYTES = 32
        self._INDEX_DTYPE = np.dtype([("shard_id", f"S{self._ID_BYTES}"), ("entries", np.int64)])

        # len cache
        self._len: Optional[int] = None

        # map from shard id to file path
        self._shard_id_map: Dict[bytes, Path] = {}

        # for fast bisect right (hot path, cache EVERYTHING)
        self._shard_ids:    Optional[List[bytes]]   = None
        self._cum_list:     Optional[List[int]]     = None
        self._starts_list:  Optional[List[int]]     = None

    @overload
    def __getitem__(self, idx: int) -> Tuple[Dict, bytes, bytes]: ...
    @overload
    def __getitem__(self, idx: slice) -> List[Tuple[Dict, bytes, bytes]]: ...

    def __getitem__(self, idx: Union[int, slice]):
        """
        Retrieve one record or a slice of records.

        Args:
            idx: Integer index or slice object.

        Returns:
            If idx is int: A tuple (data_dict, shard_id, key_bytes).
            If idx is slice: A list of such tuples.
        """
        self._ensure_index_struct()

        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            return [self[i] for i in range(start, stop, step)]

        # sanity check indices
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")

        shard_id, key = self._gid_to_shard_map(idx)

        # convert stored big-endian int64 to bytes for LMDB
        key_bytes = struct.pack(">Q", int(key))

        # load raw data
        handle = self._get_handle(shard_id)
        raw = handle.dtxn.get(key_bytes)
        if raw is None:
            raise KeyError(f"Key {key_bytes!r} not found in file {self._shard_id_map[shard_id]}")

        # unpack data
        data = msgpack.unpackb(raw, raw=False, use_list=True)

        return data, shard_id, key_bytes

    def __len__(self) -> int:
        """Return the total number of records across all shards."""
        self._ensure_index_struct()
        if self._len is None:
            self._len = int(np.sum(self._index_arr["entries"]))
        return self._len

    def __iter__(self) -> Iterator[Tuple[Dict, bytes, bytes]]:
        """
        Iterate through all records in index_map using cached transactions for speed.

        Yields:
            Tuples of (data_dict, shard_id, key_bytes) for each record.
        """
        for idx in range(len(self)):
            data, shard_id, key = self[idx]
            yield data, shard_id, key

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Ensure that all LMDB environments are closed when the reader is deleted."""
        try:
            self.close()
        except Exception:
            pass

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        # Close and drop unpicklable handles
        try:
            self.close()
        except Exception:
            pass

        state["_cache"] = OrderedDict()
        state["_attributes"] = None

        return state

    def __setstate__(self, state) -> None:
        self.__dict__.update(state)
        # Reinit caches
        if "_cache" not in self.__dict__:
            self._cache = OrderedDict()

    def _ensure_index_struct(self) -> None:
        if self._index_arr is None:
            self._index_arr = self._build_index_struct()

            diff = set(self._shard_id_map.keys()) ^ set(self._index_arr["shard_id"].tolist())
            assert diff == set(), f"Shard ID map and index struct do not match. Symmetric diffs: {diff}"

    def _gid_to_shard_map(self, gid: int) -> Tuple[bytes, int]:
        # cache minimal data for fast bisect (built once)
        if self._cum_list is None or self._starts_list is None or self._shard_ids is None:
            self._cum_list = np.cumsum(self._index_arr["entries"], dtype=np.int64).tolist()
            self._starts_list = [0] + self._cum_list[:-1]
            self._shard_ids = self._index_arr["shard_id"].tolist()

        total = self._cum_list[-1]
        if gid < 0 or gid >= total:
            raise IndexError("global id out of range")

        row = bisect_right(self._cum_list, gid)
        return self._shard_ids[row], int(gid - self._starts_list[row])

    @staticmethod
    def _open_env(path: Path) -> lmdb.Environment:
        """Open an LMDB environment with desired flags."""
        env: lmdb.Environment = lmdb.open(
            str(path),
            readonly=True,
            lock=False,
            subdir=False,
            max_dbs=2,
            readahead=True,
        )
        return env

    def _get_handle(self, shard_id: bytes) -> _Handle:
        """
        Returns a cached handle for the given file index, opening a new environment
        and transaction, and evicts the least recently used handle
        when exceeding the cache capacity.

        Args:
            shard_id (bytes): Shard ID.

        Returns:
            A _Handle object with env and txns.
        """
        path = self._shard_id_map[shard_id]
        handle = self._cache.get(path)
        if handle:
            self._cache.move_to_end(path)
            return handle

        env = self._open_env(path)

        # open transactions
        dtxn = env.begin(write=False, db=env.open_db(b"data"), buffers=True)
        atxn = env.begin(write=False, db=env.open_db(b"attr"), buffers=True)

        handle = self._Handle(env, dtxn, atxn)

        # add the newest handle to the end of the cache
        self._cache[path] = handle
        self._cache.move_to_end(path)

        self._prune_cache()

        return handle

    def _prune_cache(self) -> None:
        # remove LRU cache items if cache fills
        if len(self._cache) > self._max_open_envs:
            old_path, old_handle = self._cache.popitem(last=False)
            old_handle.close()

    def _build_index_struct(self) -> np.ndarray:
        # quick safety check
        if not self._lmdb_paths:
            raise FileNotFoundError("No LMDB files found.")

        # allocate empty array
        out = np.empty(len(self._lmdb_paths), dtype=self._INDEX_DTYPE)

        # iterate over each file, grab the length, verify contiguous indices and add to the struct
        for i, path in enumerate(self._lmdb_paths):
            # grab the env
            env = self._open_env(path)

            # first grab its shard id
            with env.begin(db=env.open_db(b"attr")) as atxn:
                shard_id_b = atxn.get("id".encode("utf-8"))
                if shard_id_b is None:
                    raise KeyError(f"Key 'id' not found in LMDB file {path}")

                _id = bytes.fromhex(msgpack.unpackb(shard_id_b, raw=False))

                if len(_id) != self._ID_BYTES:
                    raise ValueError("Shard ID length mismatch; revise dtype or normalize IDs.")

                out["shard_id"][i] = _id
                # build the shard id map
                self._shard_id_map[out["shard_id"][i]] = path

            with env.begin(db=env.open_db(b"data")) as txn:
                out["entries"][i] = txn.stat()["entries"]

            env.close()

        # sort by shard id in place
        out.sort(order="shard_id")

        return out

    def _get_attrs(self) -> Dict[bytes, Dict[str, Dict[str, Any]]]:
        self._ensure_index_struct()

        # cache isolated shard ids for fastest access
        if self._shard_ids is None:
            self._shard_ids = self._index_arr["shard_id"].tolist()

        attrs: Dict[bytes, Dict[str, Dict[str, Any]]] = {}

        # grab all attributes
        for i, shard_id in enumerate(self._shard_ids):
            path = self._shard_id_map[shard_id]
            with LMDBReader(path) as reader:
                attrs[shard_id] = reader.attrs()

        return attrs

    def attrs(self) -> Dict[bytes, Dict[str, Dict[str, Any]]]:
        if self._attributes is None:
            self._attributes = self._get_attrs()
        return self._attributes

    def close(self) -> None:
        """
        Close any cached environments and abort ongoing transactions to release resources.
        """
        while self._cache:
            _, handle = self._cache.popitem(last=False)
            handle.close()


class LMDBReader:

    def __init__(self, infile: Union[str, Path]) -> None:
        """
        Initialize a reader for a single LMDB file.

        Args:
            infile: Path to the LMDB file.
        """
        self.infile = Path(infile)
        PathValidator.is_valid_file(self.infile)

        # init the reader env
        self._env = lmdb.open(
            str(self.infile),
            readonly=True,
            lock=False,
            subdir=False,
            max_dbs=2,
            readahead=True
        )

        # create db handles
        self._data_db = self._env.open_db(b"data")
        self._attr_db = self._env.open_db(b"attr")

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        try:
            self._env.close()
        except Exception:
            pass

    def attrs(self, group: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Return attribute entries for a given group.
        """
        out: Dict[str, Dict[str, Any]] = {}

        with self._env.begin(db=self._attr_db) as txn, txn.cursor() as cursor:
            if not group:
                for k_b, v_b in cursor:
                    out[k_b.decode("utf-8", "replace")] = msgpack.unpackb(v_b, raw=False)

            else:
                v_b = txn.get(group.encode("utf-8"))
                if v_b is None:
                    raise KeyError(f"Key '{group}' not found in LMDB file {self.infile}")
                out[group] = msgpack.unpackb(v_b, raw=False)

        return out

    def to_pandas(self) -> pd.DataFrame:
        """
        Load all data records from the LMDB into a pandas DataFrame.

        Returns:
            A pandas DataFrame of all records, or an empty DataFrame if none.
        """
        records = []
        with self._env.begin(db=self._data_db) as txn, txn.cursor() as cursor:
            for _, data_packed in cursor:
                # deserialize the data
                row = msgpack.unpackb(data_packed, raw=False, use_list=True)
                records.append(row)

            # build a DataFrame from all the records
            if records:
                return pd.DataFrame.from_records(records)
            else:
                # empty LMDB
                return pd.DataFrame()
