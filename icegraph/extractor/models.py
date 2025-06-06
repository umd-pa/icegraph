# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

import os
from glob import glob

from icegraph.console import Console
from icegraph import config
from .base import Extractor

# for feature extraction implementation
from icecube.icetray import I3Tray
from icecube import dataclasses, icetray, dataio, hdfwriter
from icecube import ml_suite, phys_services
from icecube.sim_services.label_events import (
    MCLabeler,
    ClassificationConverter,
    MuonLabels
)


class FeatureExtractor(Extractor):

    """Takes in one or more .i3.zst files and runs feature extraction via the IceTray module ml_suite."""

    cls_converter = ClassificationConverter()

    def extract(self):
        Console.out(f"Running feature extraction on files ({self.path})")
        Console.spinner().start()

        tray = I3Tray()

        # Read the i3 files to memory
        tray.Add('I3Reader', Filenamelist=[
            config.PASS_2_GCD_PATH,
            *glob(os.path.join(self.path, f"*.i3.zst"))
        ])

        # This module labels MC events based on their topology
        tray.Add(
            MCLabeler,
            event_properties_name=None,
            mctree_name=config.MCTREE_NAME,
            weight_dict_name=config.WEIGHT_DICT_NAME,
            bg_mctree_name=config.BG_MCTREE_NAME
        )

        # TODO debug this, not sure why it isn't working
        # This module adds additional labels for numu events
        # tray.Add(
        #     MuonLabels,
        #     event_properties_name=None,
        #     mctree_name=config.MCTREE_NAME,
        #     weight_dict_name=config.WEIGHT_DICT_NAME
        # )

        # This module performs the feature calculation
        tray.Add(
            ml_suite.EventFeatureExtractorModule,
            cfg_file=self.config_path
        )

        # path to output file
        outfile = os.path.join(self.outdir, 'data.hdf5')

        # Serialize labels and features to hdf
        tray.AddSegment(
            hdfwriter.I3HDFWriter,
            Output=outfile,
            Keys=[
                "ml_suite_features",
                ("classification", self.cls_converter),
                "classification_emuon_entry",
                "classification_emuon_deposited"
            ],
            SubEventStreams=["InIceSplit"]
        )

        tray.Execute()

        Console.spinner().stop()

        return outfile


class TruthExtractor(Extractor):
    pass