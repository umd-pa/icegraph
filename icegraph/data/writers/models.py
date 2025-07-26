# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

import lmdb
from typing import Union, Optional
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

    def __init__(self, outfile: Optional[Union[str, Path]], mode: str = "w", map_size: int = 10 * 1024 ** 3):
        """
        Initialize the writer with a DataFrame to serialize to LMDB.

        Args:
            outfile (Optional[Union[str, Path]]): The destination path for the output file.
            mode (str): Write mode to use; "w" for write and "a" for append.
        """
        Console.out(f"Initializing LMDB writer with map_size = {map_size} bytes.")

        # call to super
        super().__init__(outfile, mode)

        # start the LMDB env
        self.env = lmdb.open(
            str(self.tmp_path) if mode == "w" else str(self.outfile),
            map_size=map_size,
            subdir=False,
            lock=True,
            readahead=False
        )

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
                new_size = self.env.info()["map_size"] * 2
                Console.out(f"LMDB map full, expanding map to {new_size} bytes.", severity=2)
                self.env.set_mapsize(new_size)
                txn = self.env.begin(write=True)

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

        try:
            with self.env.begin(write=True) as txn:
                for idx, row in enumerate(Console.progress_bar(table.itertuples(index=False), total=table.shape[0])):
                    # use 8 byte big-endian integer as the LMDB key for numeric ordering
                    key = struct.pack('>Q', idx)
                    feats = {col: getattr(row, col) for col in include_cols}

                    # serialize
                    value = msgpack.packb(feats, use_bin_type=True)
                    txn = self._safe_put(txn, key, value)

        finally:
            self.env.sync()
            self.env.close()

        if not self.tmp_path.exists():
            raise WriterError(
                f"Temporary LMDB file not found at {self.tmp_path!r}; write may have failed."
            )

        # write atomically to prevent multithread race conditions
        os.replace(str(self.tmp_path), str(self.outfile))
        Console.out(f"LMDB written to {self.outfile}")

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

        with self.env.begin(write=True) as txn:
            for i, row in enumerate(table.itertuples(index=False)):
                # use 8 byte big-endian integer as the LMDB key for numeric ordering
                key = struct.pack('>Q', start_idx + i)
                feats = {col: getattr(row, col) for col in include_cols}

                # serialize
                value = msgpack.packb(feats, use_bin_type=True)
                txn = self._safe_put(txn, key, value)

        return len(table)

    def save(self) -> None:
        """
        Manually save the file (required to call in append mode).
        """
        if not self.tmp_path.exists():
            raise WriterError(
                f"Temporary LMDB file not found at {self.tmp_path!r}; write may have failed."
            )

        # write atomically to prevent multithread race conditions
        os.replace(str(self.tmp_path), str(self.outfile))
        Console.out(f"LMDB written to {self.outfile}")

    def finish(self) -> None:
        """
        Finish the write/append op by ensuring the environment has been closed.
        """
        self.env.sync()
        self.env.close()