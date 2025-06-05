# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

import os
import pandas as pd
from glob import glob

from icegraph.console import Console
from icegraph import config
from .base import Converter

# for feature extraction implementation
from icecube.icetray import I3Tray
from icecube import dataclasses, icetray, dataio, hdfwriter
from icecube import ml_suite, phys_services
from icecube.sim_services.label_events import (
    MCLabeler,
    ClassificationConverter,
    MuonLabels
)


class HDF5ToParquet(Converter):

    @property
    def io_extensions(self):
        return ["hdf5", "parquet"]

    def convert(self):
        Console.out(f"Converting file to {self.io_extensions[1]} format")

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
            data.to_parquet(os.path.join(self.outdir, f"{file_name}.{self.io_extensions[1]}"), engine="pyarrow")

        Console.out(f"Output file(s) saved to {self.outdir}")


class I3ToHDF5(Converter):

    cls_converter = ClassificationConverter()

    @property
    def io_extensions(self):
        return ["i3.zst", "hdf5"]

    def convert(self):
        Console.out(f"Converting file to {self.io_extensions[1]} format")

        tray = I3Tray()

        # Read the i3 files to memory
        tray.Add('I3Reader', Filenamelist=[
            config.PASS_2_GCD_PATH,
            *glob(os.path.join(self.path, f"*.{self.io_extensions[0]}"))
        ])

        # This module labels MC events based on their topology
        tray.Add(
            MCLabeler,
            event_properties_name=None,
            mctree_name=config.WEIGHT_DICT_KEY,
            weight_dict_name="I3MCWeightDict",
            bg_mctree_name="BackgroundI3MCTree"
        )

        # This module adds additional labels for numu events
        # tray.Add(
        #     MuonLabels,
        #     event_properties_name=None,
        #     mctree_name=config.WEIGHT_DICT_KEY,
        #     weight_dict_name="I3MCWeightDict",
        #     bg_mctree_name="BackgroundI3MCTree"
        # )

        # This module performs the feature calculation
        tray.Add(
            ml_suite.EventFeatureExtractorModule,
            cfg_file=os.path.join(config.CONFIG_DIR, "extraction/feature_extraction.yaml")
        )

        # Serialize labels and features to hdf
        tray.AddSegment(
            hdfwriter.I3HDFWriter,
            Output=os.path.join(self.outdir, f"data.{self.io_extensions[1]}"),
            Keys=[
                "ml_suite_features",
                ("classification", self.cls_converter),
                "classification_emuon_entry",
                "classification_emuon_deposited"
            ],
            SubEventStreams=["InIceSplit"]
        )

        tray.Execute()

        Console.out(f"Output file(s) saved to {self.outdir}")