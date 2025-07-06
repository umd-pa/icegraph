# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

import lmdb
from typing import Union, Optional
import struct
from pathlib import Path
import msgpack

from .base import IGWriter
from icegraph.console import Console

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

        # use all columns if none are specified
        include_cols = include_cols or self.table.columns.tolist()

        # delete any existing files
        if outfile.exists():
            outfile.unlink()

        # initialize LMDB environment
        env = lmdb.open(
            str(outfile),
            map_size=10 * 1024 ** 3,
            subdir=False,
            lock=True,
            readahead=False
        )

        with env.begin(write=True) as txn:
            for idx, row in enumerate(Console.progress_bar(self.table.itertuples(index=False))):
                # use 8 byte big-endian integer as the LMDB key for numeric ordering
                key = struct.pack('>Q', idx)
                feats = {col: getattr(row, col) for col in include_cols}

                # serialize
                value = msgpack.packb(feats, use_bin_type=True)
                txn.put(key, value)

        env.close()