# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from dataclasses import dataclass
from datetime import datetime
import lmdb
from typing import Union, Optional, List, Dict, Self
import struct
from pathlib import Path
import msgpack
import os

import pandas as pd

from .base import IGWriter
from icegraph.console import Console
from .base.exceptions import WriterError

# allow msgpack to pack numpy objects
import msgpack_numpy as m
m.patch()

__all__ = ["LMDBWriter"]


class LMDBWriter(IGWriter):
    """
    Serializes a pandas DataFrame to LMDB format for efficient key-value storage.

    This class writes each row of the DataFrame as a serialized MessagePack object,
    using a consistent 8-byte key scheme to ensure deterministic and sortable entries.
    """

    @dataclass
    class StatisticPolicy:
        feature_cols: List[str]
        truth_cols: List[str]
        excluded_cols: List[str]

    def __init__(self, outfile: Optional[Union[str, Path]], mode: str = "w", map_size: int = 10 * 1024 ** 3, verbose: bool = True):
        """
        Initialize the writer with a DataFrame to serialize to LMDB.

        Args:
            outfile (Optional[Union[str, Path]]): The destination path for the output file.
            mode (str): Write mode to use; "w" for write and "a" for append.
            verbose (bool): Whether to print to CLI. Defaults to True. Will not filter warnings if set to False.
        """
        self._verbose = verbose
        if self._verbose:
            Console.out(f"Initializing LMDB writer with map_size = {map_size} bytes.")

        self.__version__ = 1

        # call to super
        super().__init__(outfile, mode)

        # start the LMDB env
        self._env = lmdb.open(
            str(self.tmp_path) if mode == "w" else str(self.outfile),
            map_size=map_size,
            subdir=False,
            lock=True,
            max_dbs=2,
            readahead=False
        )

        # create handles
        self._data_db = self._env.open_db(b"data", create=True)
        self._meta_db = self._env.open_db(b"meta", create=True)

    def _safe_put(self, txn, key, value) -> lmdb.Transaction:
        """
        Attempt to put the key-value pair, expanding map size if needed.
        Returns an active txn.
        """
        while True:
            try:
                txn.put(key, value)
                return txn
            except lmdb.MapFullError:
                txn.abort()
                new_size = self._env.info()["map_size"] * 2
                # always display warnings
                Console.out(f"LMDB map full, expanding map to {new_size} bytes.", severity=2)
                self._env.set_mapsize(new_size)
                txn = self._env.begin(write=True)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.finish()
        self.save()

    def write_metadata(self, metadata: Optional[Dict] = None) -> None:
        """
        Write statistics to the db file. WARNING: does not close the env automatically.

        Args:
            metadata (bool): Metadata to write to file.
        """

        def _mp(obj) -> bytes:
            return msgpack.packb(obj, use_bin_type=True)

        with self._env.begin(write=True, db=self._meta_db) as txn:
            # info
            txn.put(b"info:timestamp", _mp(datetime.now().timestamp()))
            txn.put(b"info:version", _mp(int(self.__version__)))

            if metadata:
                for category, entries in metadata.items():
                    if not isinstance(category, str):
                        raise WriterError(f"Metadata category must be a string, got {type(category)}")
                    for name, data in entries.items():
                        key = f"{category}:{name}".encode("utf-8")
                        # store the entire object (scalar/list/dict/ndarray) as one msgpack blob
                        txn.put(key, _mp(data))

    def write(self, table: pd.DataFrame, include_cols: Optional[list] = None) -> int:
        """
        Write the DataFrame to an LMDB file.

        Args:
            table (pd.DataFrame): A pandas DataFrame object to write.
            include_cols (Optional[list], optional): List of column names to include
                in the LMDB entries. If None, all columns in the table are included.

        Returns:
            int: Number of entries written.

        Raises:
            Any exceptions from LMDB or msgpack during I/O are propagated.
        """
        # use all columns if none are specified
        include_cols = include_cols or table.columns.tolist()

        with self._env.begin(write=True, db=self._data_db) as txn:
            _iter = enumerate(table.itertuples(index=False))
            if self._verbose:
                _iter = Console.progress_bar(_iter, total=table.shape[0])

            for idx, row in _iter:
                # use 8 byte big-endian integer as the LMDB key for numeric ordering
                key = struct.pack('>Q', idx)
                feats = {col: getattr(row, col) for col in include_cols}

                # serialize
                value = msgpack.packb(feats, use_bin_type=True)
                txn = self._safe_put(txn, key, value)

        return len(table)

    def append(self, table: pd.DataFrame, start_idx: int, include_cols: Optional[list] = None) -> int:
        """
        Append the DataFrame to an existing LMDB file, starting keys from `start_idx`. Must call save() to save the
        file after finishing appends.

        Args:
            table (pd.DataFrame): A pandas DataFrame object to write.
            start_idx (int): The starting index for new entries.
            include_cols (Optional[list]): Columns to include. If None, use all.

        Returns:
            int: Number of entries written.
        """
        include_cols = include_cols or table.columns.tolist()

        with self._env.begin(write=True, db=self._data_db) as txn:
            for idx, row in enumerate(table.itertuples(index=False)):
                # use 8 byte big-endian integer as the LMDB key for numeric ordering
                key = struct.pack('>Q', start_idx + idx)
                feats = {col: getattr(row, col) for col in include_cols}

                # serialize
                value = msgpack.packb(feats, use_bin_type=True)
                txn = self._safe_put(txn, key, value)

        return len(table)

    def save(self) -> None:
        """
        Manually save the file.
        """
        if self.mode == "a":
            # no op on append
            return

        if not self.tmp_path.exists():
            raise WriterError(
                f"Temporary LMDB file not found at {self.tmp_path!r}; write may have failed."
            )

        # write atomically to prevent multithread race conditions
        os.replace(str(self.tmp_path), str(self.outfile))
        if self._verbose:
            Console.out(f"LMDB written to {self.outfile}")

    def finish(self) -> None:
        """
        Finish the write/append op by ensuring the environment has been closed.
        """
        self._env.sync()
        self._env.close()
