# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from dataclasses import dataclass
import lmdb
from pathlib import Path
from typing import List, Tuple, Union, Sequence, Iterator, Dict, Optional, overload, Self, Any, ClassVar, Generator
from collections import OrderedDict
import msgpack
import hashlib
import threading

import pandas as pd
import numpy as np

from icegraph.config import IGConfig
from icegraph.pathutils import PathValidator, PathResolver
from icegraph.console import Console
from icegraph.utils import Statistics

import msgpack_numpy as m
m.patch()  # allow msgpack to work with numpy objects

__all__ = ["LMDBDatasetShardReader", "LMDBReader"]


class LMDBDatasetShardReader:
    """
    Provides access to one or more LMDB files from the same dataset written using 8-byte big endian integer keys.
    Source LMDB files are pre-set on configuration and are used on all instances of this class.
    """

    _lmdb_paths:        ClassVar[Optional[Tuple[Path, ...]]]    = None
    _index_arr:         ClassVar[Optional[np.ndarray]]          = None
    _max_open_envs:     ClassVar[Optional[int]]                 = None

    # one structured dtype
    _INDEX_DTYPE: ClassVar[np.dtype] = np.dtype([("file_index", np.int32), ("key", ">u8")])

    @dataclass
    class _Handle:
        env: lmdb.Environment
        dtxn: lmdb.Transaction
        atxn: lmdb.Transaction

    def __init__(self) -> None:
        """
        Initialize the shard reader.
        """
        # cache for open env/txn
        self._cache: "OrderedDict[Path, LMDBDatasetShardReader._Handle]" = OrderedDict()

        # only allow instantiation after configure
        if type(self)._lmdb_paths is None:
            raise RuntimeError("Reader not configured; call configure() first.")

        # attributes cache
        self._attributes: Optional[Dict[int, Dict[str, Dict[str, Any]]]] = None

    @overload
    def __getitem__(self, idx: int) -> Tuple[Dict, int, bytes]: ...
    @overload
    def __getitem__(self, idx: slice) -> List[Tuple[Dict, int, bytes]]: ...

    def __getitem__(self, idx: Union[int, slice]):
        """
        Retrieve one record or a slice of records.

        Args:
            idx: Integer index or slice object.

        Returns:
            If idx is int: A tuple (data_dict, file_index, key_bytes).
            If idx is slice: A list of such tuples.
        """
        cls = type(self)

        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            return [self[i] for i in range(start, stop, step)]

        # sanity check indices
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")

        rec = cls._index_arr[idx]
        file_index = int(rec["file_index"])
        # convert stored big-endian int64 to bytes for LMDB
        key_bytes = int(rec["key"]).to_bytes(8, "big", signed=False)

        # load raw data
        handle = self._get_handle(file_index)
        raw = handle.dtxn.get(key_bytes)
        if raw is None:
            raise KeyError(f"Key {key_bytes!r} not found in file {cls._lmdb_paths[file_index]}")

        # unpack data
        data = msgpack.unpackb(raw, raw=False, use_list=True)

        return data, file_index, key_bytes

    def __len__(self) -> int:
        """Return the total number of records across all shards."""
        return int(type(self)._index_arr.shape[0])

    def __iter__(self) -> Iterator[Tuple[Dict, int, bytes]]:
        """
        Iterate through all records in index_map using cached transactions for speed.

        Yields:
            Tuples of (data_dict, file_index, key_bytes) for each record.
        """
        for idx in range(len(self)):
            data, file_index, key = self[idx]
            yield data, file_index, key

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Ensure that all LMDB environments are closed when the reader is deleted."""
        try:
            self.close()
        except Exception:
            pass

    @staticmethod
    def _open_env(path: Path) -> lmdb.Environment:
        """Open an LMDB environment with desired flags."""
        return lmdb.open(
            str(path),
            readonly=True,
            lock=False,
            subdir=False,
            max_dbs=2,
            readahead=True,
        )

    def _get_handle(self, file_index: int) -> _Handle:
        """
        Returns a cached handle for the given file index, opening a new environment
        and transaction, and evicts the least recently used handle
        when exceeding the cache capacity.

        Args:
            file_index: Index into self.lmdb_paths.

        Returns:
            A _Handle object with env and txn.
        """
        cls = type(self)

        if cls._lmdb_paths is None:
            raise RuntimeError("Reader not configured; call configure() first.")

        path = cls._lmdb_paths[file_index]
        handle = self._cache.get(path)
        if handle:
            self._cache.move_to_end(path)
            return handle

        env = self._open_env(path)

        # open transactions
        dtxn = env.begin(write=False, db=env.open_db(b"data"))
        atxn = env.begin(write=False, db=env.open_db(b"attr"))

        handle = self._Handle(env, dtxn, atxn)

        self._cache[path] = handle
        self._cache.move_to_end(path)

        if len(self._cache) > cls._max_open_envs:
            old_path, old_handle = self._cache.popitem(last=False)
            for t in (old_handle.atxn, old_handle.dtxn):
                try:
                    t.abort()
                except Exception:
                    pass
            try:
                old_handle.env.close()
            except Exception:
                pass
        return handle

    @classmethod
    def _build_index_struct(cls) -> np.ndarray:
        if not cls._lmdb_paths:
            raise FileNotFoundError("No LMDB files found.")
        Console.out("Indexing LMDB files (building key map)...")

        rows_fi: List[int] = []
        rows_key: List[int] = []

        for fi, path in Console.progress_bar(list(enumerate(cls._lmdb_paths))):
            env = cls._open_env(path)
            with env.begin(db=env.open_db(b"data")) as txn:
                entries = txn.stat()["entries"]
                for i in range(entries):
                    rows_fi.append(fi)
                    rows_key.append(i)
            env.close()

        n = len(rows_fi)
        out = np.empty(n, dtype=cls._INDEX_DTYPE)
        if n:
            out["file_index"] = np.asarray(rows_fi, dtype=np.int32)
            out["key"] = np.asarray(rows_key, dtype=">i8")
        return out

    def _get_attrs(self) -> Dict[int, Dict[str, Dict[str, Any]]]:
        attrs: Dict[int, Dict[str, Dict[str, Any]]] = {}

        # grab all attributes
        for fi, path in enumerate(type(self)._lmdb_paths):
            with LMDBReader(path) as reader:
                attrs[fi] = reader.attrs()

        return attrs

    def attrs(self) -> Dict[int, Dict[str, Dict[str, Any]]]:
        if self._attributes is None:
            self._attributes = self._get_attrs()
        return self._attributes

    @classmethod
    def configure(
        cls,
        source: Union[str, Path, Sequence[Union[str, Path]]],
        max_open_envs: int = 4,
        clean: bool = False
    ) -> None:
        """
        Pre-configure the shard reader.

        Args:
            source: Path or sequence of paths to LMDB files or a directory containing LMDB files.
            max_open_envs: Maximum number of LMDB environments to keep open concurrently.
            clean (bool): Whether to reset the configuration with new values.
        """
        # clean old configs if required
        if clean:
            cls._max_open_envs = None
            cls._lmdb_paths = None

        if cls._max_open_envs is None:
            cls._max_open_envs = max_open_envs

        # normalize sources and store
        new_paths: Tuple[Path] = tuple(PathResolver.normalize_sources(source, ".lmdb"))
        if cls._lmdb_paths is None:
            cls._lmdb_paths = new_paths
        elif cls._lmdb_paths != new_paths:
            raise RuntimeError("Reader already configured for a different source.")

        cls._index_arr = cls._build_index_struct()

    def close(self) -> None:
        """
        Close any cached environments and abort ongoing transactions to release resources.
        """
        while self._cache:
            _, handle = self._cache.popitem(last=False)
            for t in (handle.atxn, handle.dtxn):
                try:
                    t.abort()
                except Exception:
                    pass
            try:
                handle.env.close()
            except Exception:
                pass


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
