# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

import os
import pandas as pd

from icegraph.console import Console
from icegraph import config
from .base import Converter

# for feature extraction implementation
from icecube import ml_suite


class H5ToParquet(Converter):

    @property
    def to_filetype(self) -> str:
        return "parquet"

    def convert(self) -> None:
        """Convert an HDF5 file to a collection of Parquet files for fast training.
        Returns ``None``.
        """
        Console.out(f"Converting file to {self.to_filetype} format")

        pulse_chunks = []
        truth_chunks = []

        # count total rows for progress bar
        store = pd.HDFStore(self.path, mode="r")
        pulse_nrows, truth_nrows = [
            store.get_storer(key).nrows for key in [
                config.PULSE_SERIES, config.TRUTH_TABLE
            ]
        ]
        store.close()

        # load chunk by chunk
        for start in Console.prog(range(0, pulse_nrows, config.CHUNK_SIZE)):
            chunk = pd.read_hdf(
                self.path, key=config.PULSE_SERIES, start=start, stop=start + config.CHUNK_SIZE
            )
            pulse_chunks.append(chunk)

        for start in Console.prog(range(0, truth_nrows, config.CHUNK_SIZE)):
            chunk = pd.read_hdf(
                self.path, key=config.TRUTH_TABLE, start=start, stop=start + config.CHUNK_SIZE
            )
            truth_chunks.append(chunk)

        # merge chunks to single dataset, save to files
        for chunks, file_name in [(pulse_chunks, "pulse_series"), (truth_chunks, "truth")]:
            data = pd.concat(chunks, ignore_index=True)
            data.to_parquet(os.path.join(self.outdir, f"{file_name}.{self.to_filetype}"), engine="pyarrow")

        Console.out(f"Output files saved to {self.outdir}")


class I3ToH5(Converter):

    @property
    def to_filetype(self):
        return "hdf5"

    def convert(self):
        # TODO - implement feature extraction / file-type conversion using icecube.ml_suite
        pass