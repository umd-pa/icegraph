# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

import msgpack
import lmdb
from typing import Optional, Union
from pathlib import Path
import subprocess

import pandas as pd

from .base import IGMerger
from icegraph.console import Console
from icegraph.data.writers import LMDBWriter
from icegraph.pathutils import PathResolver

__all__ = ["HDF5Merger", "LMDBMerger"]


class HDF5Merger(IGMerger):
    """
    Handles merging of multiple HDF5 files into a single output file using the
    `hdfwriter-merge` CLI tool.

    Attributes:
        file_ext (str): File extension for HDF5 files.
    """

    file_ext = "hdf5"

    def merge(self, outfile: Optional[Union[str, Path]] = None) -> Path:
        """
        Merge multiple HDF5 files into a single HDF5 output file.

        Returns:
            Path: Path to the merged output file.
        """
        Console.banner("HDF5 Merger")
        Console.out(f"Merging {len(self.files)} HDF5 files...")
        Console.spinner().start()

        resolver = PathResolver(path=outfile, origin=self.input_dir, extension="hdf5", stage="merger")
        outfile = resolver.resolve()

        # configure the merge command
        merge_command = ["hdfwriter-merge", "-o", str(outfile)] + [str(f) for f in self.files]

        # run merge
        try:
            subprocess.run(merge_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        except subprocess.CalledProcessError as e:
            Console.spinner().stop()
            raise

        Console.spinner().stop()
        Console.out(f"Merge complete, output file saved to {outfile}")

        return outfile


class LMDBMerger(IGMerger):
    """
    Handles merging of multiple LMDB files into a single output file.

    Attributes:
        file_ext (str): File extension for LMDB files.
    """

    file_ext = "lmdb"

    def merge(self, outfile: Optional[Union[str, Path]] = None) -> Path:
        """
        Merge multiple LMDB files into a single LMDB output file.

        Returns:
            Path: Path to the merged output file.
        """
        Console.banner("LMDB Merger")
        Console.out(f"Merging {len(self.files)} LMDB files...")

        resolver = PathResolver(path=outfile, origin=self.input_dir, extension="lmdb", stage="merger")
        outfile = resolver.resolve()

        global_idx = 0

        for path in Console.progress_bar(self.files):
            # Read LMDB entries and decode
            rows = []
            with lmdb.open(str(path), readonly=True, lock=False, readahead=False, subdir=False) as env:
                with env.begin() as txn:
                    cursor = txn.cursor()
                    for _, value in cursor:
                        rows.append(msgpack.unpackb(value, raw=False))

            # Append to output
            writer = LMDBWriter(pd.DataFrame(rows))
            written = writer.append(outfile, start_idx=global_idx)
            global_idx += written

        Console.out(f"Merge complete, output file saved to {outfile}")
        return outfile