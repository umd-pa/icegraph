# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

import lmdb
import uuid
from typing import Union, Optional
import struct
from pathlib import Path
import msgpack
import os

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

    Attributes:
        table (pd.DataFrame): The data table to be serialized and stored.
    """

    @staticmethod
    def _safe_put(txn, key, value, env) -> lmdb.Transaction:
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
                new_size = env.info()["map_size"] * 2
                Console.out(f"LMDB map full, expanding map to {new_size} bytes.", severity=2)
                env.set_mapsize(new_size)
                txn = env.begin(write=True)

    def write(self, outfile: Union[str, Path], include_cols: Optional[list] = None) -> None:
        """
        Write the DataFrame to an LMDB file.

        Args:
            outfile (Union[str, Path]): Path to the output LMDB file.
            include_cols (Optional[list], optional): List of column names to include
                in the LMDB entries. If None, all columns in the table are included.

        Raises:
            Any exceptions from LMDB or msgpack during I/O are propagated.
        """
        outfile = Path(outfile)
        outfile.parent.mkdir(parents=True, exist_ok=True)

        tmp_path = outfile.parent / f".{outfile.name}.{uuid.uuid4().hex}.tmp"
        if tmp_path.exists():
            tmp_path.unlink()

        # use all columns if none are specified
        include_cols = include_cols or self.table.columns.tolist()

        # delete any existing files
        if outfile.exists():
            outfile.unlink()

        # initialize LMDB environment
        env = lmdb.open(
            str(tmp_path),
            map_size=10 * 1024 ** 3,
            subdir=False,
            lock=True,
            readahead=False
        )

        try:
            with env.begin(write=True) as txn:
                for idx, row in enumerate(Console.progress_bar(self.table.itertuples(index=False), total=self.table.shape[0])):
                    # use 8 byte big-endian integer as the LMDB key for numeric ordering
                    key = struct.pack('>Q', idx)
                    feats = {col: getattr(row, col) for col in include_cols}

                    # serialize
                    value = msgpack.packb(feats, use_bin_type=True)
                    txn = self._safe_put(txn, key, value, env)

        finally:
            env.close()

        if not tmp_path.exists():
            raise WriterError(
                f"Temporary LMDB file not found at {tmp_path!r}; write may have failed."
            )

        # write atomically to prevent multithread race conditions
        os.replace(str(tmp_path), str(outfile))
        Console.out(f"LMDB written to {outfile}")

        env.close()

    def append(self, outfile: Union[str, Path], start_idx: int, include_cols: Optional[list] = None) -> int:
        """
        Append the DataFrame to an existing LMDB file, starting keys from `start_idx`.

        Args:
            outfile (Union[str, Path]): Path to the LMDB file to append to.
            start_idx (int): The starting index for new entries.
            include_cols (Optional[list]): Columns to include. If None, use all.

        Returns:
            int: Number of entries written.
        """
        outfile = Path(outfile)
        outfile.parent.mkdir(parents=True, exist_ok=True)

        include_cols = include_cols or self.table.columns.tolist()

        env = lmdb.open(
            str(outfile),
            map_size=10 * 1024 ** 3,
            subdir=False,
            lock=True,
            readahead=False
        )

        with env.begin(write=True) as txn:
            for i, row in enumerate(self.table.itertuples(index=False)):
                key = struct.pack('>Q', start_idx + i)
                feats = {col: getattr(row, col) for col in include_cols}
                value = msgpack.packb(feats, use_bin_type=True)
                txn = self._safe_put(txn, key, value, env)

        env.close()
        return len(self.table)