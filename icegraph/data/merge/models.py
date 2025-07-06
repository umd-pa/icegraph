# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import Union
import os
from pathlib import Path
import subprocess

from .base import IGMerger
from icegraph.console import Console


class HDF5Merger(IGMerger):
    """
    Handles merging of multiple HDF5 files into a single output file using the
    `hdfwriter-merge` CLI tool.

    Attributes:
        file_ext (str): File extension for HDF5 files.
    """

    file_ext = "hdf5"

    def merge(self) -> Path:
        """
        Merge multiple HDF5 files into a single HDF5 output file.

        Returns:
            Path: Path to the merged output file.
        """
        Console.out(f"Merging {len(self.files)} HDF5 files...")
        Console.spinner().start()

        # configure the merge command
        merge_command = [
            "hdfwriter-merge", "-o", self.output_file
        ] + self.files

        # run merge
        subprocess.run(
            merge_command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )

        Console.spinner().stop()
        Console.out(f"Merge complete, output file saved to {self.output_file}")

        return self.output_file


class LMDBMerger(IGMerger):
    """
    Placeholder class for merging multiple LMDB files into one.

    Attributes:
        file_ext (str): File extension for LMDB files.
    """

    file_ext = "lmdb"

    def merge(self) -> Path:
        """
        Merge multiple LMDB files into a single LMDB output file.

        Returns:
            Path: Path to the merged output file.

        Raises:
            NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError("Merging LMDB files is not yet implemented.")