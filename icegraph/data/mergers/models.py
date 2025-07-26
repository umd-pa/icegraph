# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

import msgpack
import lmdb
from typing import Optional, Union
from pathlib import Path
import subprocess
import math

import pandas as pd

from .base import IGMerger
from icegraph.console import Console
from icegraph.data.writers import LMDBWriter
from icegraph.pathutils import PathResolver
from .base.exceptions import MissingLMDBFilesError, MissingHDF5FilesError, MergeError, MergeToolNotFoundError

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

        Raises:
            MergeToolNotFoundError, MergeError, MissingHDF5FilesError
        """
        Console.banner("HDF5 Merger")

        if len(self.files) == 0:
            raise MissingHDF5FilesError(f"No HDF5 files found in directory {self.indir!s}")

        Console.out(f"Merging {len(self.files)} HDF5 files...")

        with Console.spinner():
            resolver = PathResolver(path=outfile, origin=self.indir, extension="hdf5", stage="merger")
            outfile = resolver.resolve()

            # configure the merge command
            merge_command = ["hdfwriter-merge", "-o", str(outfile)] + [str(f) for f in self.files]
            try:
                subprocess.run(merge_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            except FileNotFoundError:
                Console.spinner().stop()
                raise MergeToolNotFoundError("Could not find `hdfwriter-merge` on $PATH")
            except subprocess.CalledProcessError as e:
                Console.spinner().stop()
                stderr = e.stderr.decode(errors="ignore").strip()
                raise MergeError(f"`hdfwriter-merge` failed: {stderr!r}") from e

        Console.out(f"Merge complete, output file saved to {outfile}")

        return outfile


class LMDBMerger(IGMerger):
    """
    Handles merging of multiple LMDB files into a single output file.

    Attributes:
        file_ext (str): File extension for LMDB files.

    Raises:
        MergeToolNotFoundError, MergeError, MissingLMDBFilesError
    """

    file_ext = "lmdb"

    def merge(self, outfile: Optional[Union[str, Path]] = None) -> Path:
        """
        Merge multiple LMDB files into a single LMDB output file.

        Returns:
            Path: Path to the merged output file.
        """
        Console.banner("LMDB Merger")

        if len(self.files) == 0:
            raise MissingLMDBFilesError(f"No LMDB files found in directory {self.indir!s}")

        Console.out(f"Merging {len(self.files)} LMDB files...")

        resolver = PathResolver(path=outfile, origin=self.indir, extension="lmdb", stage="merger")
        outfile = resolver.resolve()

        # guess total file size to set map_size in writer
        total_file_size = sum(
            file.stat().st_size
            for file in self.files
            if file.is_file()
        )
        map_size = 1 << math.ceil(math.log2(total_file_size * 1.3))

        global_idx = 0
        writer = LMDBWriter(outfile, map_size=map_size)

        for src in Console.progress_bar(self.files):
            # open source LMDB
            with lmdb.open(str(src),
                           subdir=False, readonly=True,
                           lock=False, readahead=False,
                           meminit=False) as env:
                with env.begin(write=False) as txn:
                    rows: list[dict] = []
                    for _, raw in txn.cursor():
                        try:
                            rows.append(msgpack.unpackb(raw, raw=False))
                        except Exception as e:
                            raise MergeError(
                                f"Corrupt record in {src}: {e}"
                            )

            # convert to DataFrame (may be large!)
            df = pd.DataFrame(rows)
            if df.empty:
                Console.out(f"No records in {src}", severity=2)
                continue

            written = writer.append(df, start_idx=global_idx)
            global_idx += written

        # shutdown the writer environment and save the file
        writer.finish()
        writer.save()

        Console.out(f"Merge complete, output file saved to {outfile}")
        return outfile