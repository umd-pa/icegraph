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

__all__ = ["LMDBConfiguredShardReader", "LMDBReader"]


class LMDBConfiguredShardReader:
    """
    Provides access to one or more LMDB files from the same dataset written using 8-byte big endian integer keys.
    Source LMDB files are pre-set on configuration and are used on all instances of this class.
    """

    _lmdb_paths:        ClassVar[Optional[Tuple[Path, ...]]]    = None
    _index_path:        ClassVar[Optional[Path]]                = None
    _index_arr:         ClassVar[Optional[np.ndarray]]          = None
    _max_open_envs:     ClassVar[Optional[int]]                 = None
    _index_lock:        ClassVar[threading.Lock]                = threading.Lock()

    # one structured dtype for compact storage
    _INDEX_DTYPE: ClassVar[np.dtype] = np.dtype([("file_index", np.int32), ("key", ">u8")])

    @dataclass
    class _Handle:
        env: lmdb.Environment
        dtxn: lmdb.Transaction
        mtxn: Optional[lmdb.Transaction]

    def __init__(self) -> None:
        """
        Initialize the shard reader.
        """
        # cache for open env/txn
        self._cache: "OrderedDict[Path, LMDBConfiguredShardReader._Handle]" = OrderedDict()

        # only allow instantiation after configure
        if type(self)._lmdb_paths is None:
            raise RuntimeError("Reader not configured; call configure() first.")

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
        cls._load_index_memmap()

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
        type(self)._load_index_memmap()
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

    @classmethod
    def _load_index_memmap(cls) -> None:
        if cls._index_arr is None:
            with cls._index_lock:
                if cls._index_arr is None:
                    if cls._index_path is None:
                        raise RuntimeError("Index memmap path is not set. Call configure() first.")

                    arr = np.load(cls._index_path, mmap_mode="r")
                    if arr.dtype != cls._INDEX_DTYPE:
                        raise TypeError(f"Unexpected index dtype: {arr.dtype}")

                    cls._index_arr = arr

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

        # open the data txn
        dtxn = env.begin(write=False, db=env.open_db(b"data"))

        # open the metadata txn
        try:
            meta_db = env.open_db(b"meta")
            mtxn = env.begin(write=False, db=meta_db)
        except lmdb.NotFoundError:
            mtxn = None

        handle = self._Handle(env, dtxn, mtxn)

        self._cache[path] = handle
        self._cache.move_to_end(path)

        if len(self._cache) > cls._max_open_envs:
            old_path, old_handle = self._cache.popitem(last=False)
            for t in (old_handle.mtxn, old_handle.dtxn):
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
    def _build_index_struct_from_scan(cls) -> np.ndarray:
        if not cls._lmdb_paths:
            raise FileNotFoundError("No LMDB files found.")
        Console.out("Indexing LMDB files (building key map)...")

        rows_fi: List[int] = []
        rows_key: List[int] = []

        for fi, path in Console.progress_bar(list(enumerate(cls._lmdb_paths))):
            env = cls._open_env(path)
            with env.begin(db=env.open_db(b"data")) as txn:
                cur = txn.cursor()
                for k, _ in cur:
                    # keys are 8 bytes big-endian signed; store as int64 big-endian
                    rows_fi.append(fi)
                    rows_key.append(int.from_bytes(k, "big", signed=False))
            env.close()

        n = len(rows_fi)
        out = np.empty(n, dtype=cls._INDEX_DTYPE)
        if n:
            out["file_index"] = np.asarray(rows_fi, dtype=np.int32)
            out["key"] = np.asarray(rows_key, dtype=">i8")
        return out

    @classmethod
    def _build_index_struct_from_df(
            cls,
            map_df: pd.DataFrame,
            file_index_column: str = "file_index",
            key_column: str = "key",
            **_
    ) -> np.ndarray:
        # file_index convert to int32
        file_index = map_df[file_index_column].to_numpy(dtype=np.int32, copy=False)

        # keys
        keys_list = map_df[key_column].values.tolist()
        if any(len(b) != 8 for b in keys_list):
            raise ValueError("All keys must be 8 bytes.")

        keys = np.asarray(map_df[key_column].values, dtype="S8")  # fixed-size 8 bytes
        key_arr = np.frombuffer(keys.data, dtype=">i8", count=len(keys))

        out = np.empty(len(keys), dtype=cls._INDEX_DTYPE)
        out["file_index"] = file_index
        out["key"] = key_arr
        return out

    @property
    def stats(self) -> Tuple[Statistics, Statistics]:
        """
        Return global dataset statistics merged across all LMDB shards.

        Returns:
            Tuple[Statistics, Statistics]: Returns a tuple with feature stats and truth stats, in that order.
        """
        paths = type(self)._lmdb_paths
        if not paths:
            raise RuntimeError("Reader not configured or no LMDB files found. Call configure() first.")

        def iter_shard_stats() -> Generator[Tuple[Statistics, Statistics], Any, None]:
            for p in Console.progress_bar(paths, total=len(paths)):
                try:
                    with LMDBReader(p) as lmdb_file:
                        f_stats_dict = lmdb_file.metadata("f_stats")
                        t_stats_dict = lmdb_file.metadata("t_stats")
                        yield Statistics.from_dict(f_stats_dict), Statistics.from_dict(t_stats_dict)
                except (lmdb.NotFoundError, FileNotFoundError, KeyError) as e:
                    Console.out(f"Skipping {p}: no stats found ({e}).", severity=2)

        Console.out("Collecting dataset metadata and computing global statistics...")

        global_f: Optional[Statistics] = None
        global_t: Optional[Statistics] = None

        for f_stat, t_stat in iter_shard_stats():
            global_f = f_stat if global_f is None else global_f.merge(f_stat)
            global_t = t_stat if global_t is None else global_t.merge(t_stat)

        if global_f is None or global_t is None:
            raise RuntimeError("No shard statistics found; cannot compute globals.")

        return global_f, global_t

    @classmethod
    def configure(
        cls,
        source: Union[str, Path, Sequence[Union[str, Path]]],
        max_open_envs: int = 4,
        map_df: Optional[pd.DataFrame] = None,
        clean: bool = False,
        **kwargs
    ) -> None:
        """
        Pre-configure the shard reader.

        Args:
            source: Path or sequence of paths to LMDB files or a directory containing LMDB files.
            max_open_envs: Maximum number of LMDB environments to keep open concurrently.
            map_df: Optional prebuilt mapping DataFrame with columns ('file_index', 'key') to use for indexing.
            clean (bool): Whether to reset the configuration with new values.
            **kwargs: Additional arguments for index initialization when using map_df.
        """
        cache_dir = IGConfig.get_xdg_cache_dir()

        # clean old configs if required
        if clean:
            cls._max_open_envs = None
            cls._lmdb_paths = None
            cls._index_path = None
            cls._index_arr = None

        # store max open env
        if cls._max_open_envs is None:
            cls._max_open_envs = max_open_envs

        # normalize sources and store
        new_paths: Tuple[Path] = tuple(PathResolver.normalize_sources(source, ".lmdb"))
        if cls._lmdb_paths is None:
            cls._lmdb_paths = new_paths
        elif cls._lmdb_paths != new_paths:
            raise RuntimeError("Reader already configured for a different source.")

        # build index map and store
        if cls._index_path is None:
            h = hashlib.sha1(("|".join(map(str, cls._lmdb_paths))).encode()).hexdigest()[:12]
            index_path = cache_dir / f"lmdb_index_{h}.npy"

            cls._index_path = Path(index_path).resolve()
            cls._index_path.parent.mkdir(parents=True, exist_ok=True)

            # build the index map
            if map_df is None:
                arr = cls._build_index_struct_from_scan()
            else:
                arr = cls._build_index_struct_from_df(map_df, **kwargs)

            # atomically write the array to disk
            tmp = cls._index_path.with_suffix(".npy.tmp")

            with open(tmp, "wb") as f:
                np.save(f, arr, allow_pickle=False)

            tmp.replace(cls._index_path)

            # force reload via memmap in each process
            cls._index_arr = None

    def close(self) -> None:
        """
        Close any cached environments and abort ongoing transactions to release resources.
        """
        while self._cache:
            _, handle = self._cache.popitem(last=False)
            for t in (handle.mtxn, handle.dtxn):
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
        self._meta_db = self._env.open_db(b"meta")

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        try:
            self._env.close()
        except Exception:
            pass

    def metadata(self, category: str) -> Dict:
        prefix = f"{category}:".encode("utf-8")
        data: Dict[str, Any] = {}
        with self._env.begin(db=self._meta_db) as txn, txn.cursor() as cursor:
            for key_bytes, value_packed in cursor:
                if not key_bytes.startswith(prefix):
                    continue

                key = key_bytes[len(prefix):].decode("utf-8", "replace")

                # append to rows
                data[key] = msgpack.unpackb(value_packed, raw=False)

        return data

    def to_pandas(self) -> pd.DataFrame:
        """
        Load all data records from the LMDB into a pandas DataFrame.

        Assumes each value is a msgpack-packed dict with at least an 'index' key.
        Rows are sorted by the 'index' column.

        Returns:
            A pandas DataFrame of all records, or an empty DataFrame if none.
        """
        # this whole method is stupid, will fix later
        records = []
        with self._env.begin(db=self._data_db) as txn, txn.cursor() as cursor:
            for _, data_packed in cursor:
                # deserialize the data
                row = msgpack.unpackb(data_packed, raw=False, use_list=True)
                records.append(row)

            # build a DataFrame from all the records
            if records:
                return pd.DataFrame.from_records(records).sort_values(by="index").reset_index(drop=True)
            else:
                # empty LMDB
                return pd.DataFrame()
