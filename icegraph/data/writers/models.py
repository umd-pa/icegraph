# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from datetime import datetime
import lmdb
from typing import Union, Optional, Dict, Self, Iterable, NamedTuple, Any
import struct
from pathlib import Path
import msgpack
import os

import pandas as pd

from .base import Writer
from icegraph.console import Console
from .base.exceptions import WriterError
from icegraph._version import __version__

# allow msgpack to pack numpy objects
import msgpack_numpy as m
m.patch()

__all__ = ["LMDBWriter"]


class LMDBWriter(Writer):
    """
    Serializes one or more pandas DataFrame(s) to LMDB format for efficient key-value storage.

    This class writes each row of the DataFrame as a serialized MessagePack object,
    using a consistent 8-byte key scheme to ensure deterministic and sortable entries.
    """

    suffix: str = ".lmdb"

    def __init__(self, outfile: Optional[Union[str, Path]], map_size: int = 10 * 1024 ** 3):
        """
        Initialize the writer with a DataFrame to serialize to LMDB.

        Args:
            outfile (Optional[Union[str, Path]]): The destination path for the output file.
            map_size (int): The map size in bytes. Expand if necessary.
        """

        # call to super
        super().__init__(outfile)

        # start the LMDB env
        self._env = lmdb.open(
            str(self.tmp_path),
            map_size=map_size,
            subdir=False,
            lock=True,
            max_dbs=2,
            readahead=False
        )

        # create handles
        self._data_db = self._env.open_db(b"data", create=True)
        self._attr_db = self._env.open_db(b"attr", create=True)

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

    def write_attrs(self, groups: Dict[str, Dict[str, Any]], include_defaults: bool = True) -> None:
        """
        Write attributes to the db file.

        Args:
            groups (bool): Attribute groups to write to file.
            include_defaults (bool): Whether to write default metadata to the file, like timestamp and software version.
        """

        def _mp(obj) -> bytes:
            return msgpack.packb(obj, use_bin_type=True)

        def _write_defaults(_txn: lmdb.Transaction) -> None:
            """Write default info to the file."""
            _txn.put(
                "info".encode("utf-8"),
                _mp({
                    "timestamp": datetime.now().timestamp(),
                    "version": __version__
                })
            )

        with self._env.begin(write=True, db=self._attr_db) as txn:
            # info
            if include_defaults:
                _write_defaults(txn)

            for group, data in groups.items():
                # store the entire object (scalar/list/dict/ndarray) as one msgpack blob
                txn.put(f"{group}".encode("utf-8"), _mp(data))

    def write(self, df: pd.DataFrame) -> None:
        """
        Write the DataFrame to an LMDB file.

        Args:
            df (pd.DataFrame): A pandas DataFrame object to write.

        Raises:
            Any exceptions from LMDB or msgpack during I/O are propagated.
        """
        # type hinting for linters
        row: NamedTuple

        with self._env.begin(write=True, db=self._data_db) as txn:
            for idx, row in  enumerate(df.itertuples(index=False)):
                # use 8 byte big-endian integer as the LMDB key for numeric ordering
                key = struct.pack('>Q', idx)

                # serialize
                value = msgpack.packb(row._asdict(), use_bin_type=True)
                txn = self._safe_put(txn, key, value)

    def write_iterable(self, iter_dfs: Iterable[pd.DataFrame]) -> None:
        """
        Write multiple DataFrames to an LMDB file.

        Args:
            iter_dfs (Iterable[pd.DataFrame]): An iterable of pandas DataFrame objects to write.

        Raises:
            Any exceptions from LMDB or msgpack during I/O are propagated.
        """
        # init write index
        idx = 0

        # type hinting for linters
        row: NamedTuple

        with self._env.begin(write=True, db=self._data_db) as txn:
            for df in iter_dfs:
                for row in df.itertuples(index=False):
                    # use 8 byte big-endian integer as the LMDB key for numeric ordering
                    key = struct.pack('>Q', idx)

                    # serialize
                    value = msgpack.packb(row._asdict(), use_bin_type=True)
                    txn = self._safe_put(txn, key, value)

                    # increment idx
                    idx += 1

    def save(self) -> None:
        """
        Save the file.
        """
        if not self.tmp_path.exists():
            raise WriterError(
                f"Temporary LMDB file not found at {self.tmp_path!r}; write may have failed."
            )

        # write atomically to prevent multithread race conditions
        os.replace(str(self.tmp_path), str(self.outfile))

    def finish(self) -> None:
        """
        Finish the write op by ensuring the environment has been closed.
        """
        self._env.sync()
        self._env.close()
