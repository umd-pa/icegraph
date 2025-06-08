# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from pathlib import Path

from icegraph.console import Console
from .base import Extractor

# have to wrap in try/except block so sphinx can properly generate docs
try:
    from icecube.icetray import I3Tray
    from icecube import dataclasses, icetray, dataio, hdfwriter, ml_suite
    from icecube.sim_services.label_events import (
        MCLabeler,
        ClassificationConverter,
        MuonLabels
    )
except ImportError:
    I3Tray = None
    dataclasses = None
    icetray = None
    dataio = None
    hdfwriter = None
    ml_suite = None
    MCLabeler = None
    ClassificationConverter = None
    MuonLabels = None


__all__ = ["FeatureExtractor"]

class FeatureExtractor(Extractor):
    """
    Extracts features from .i3.zst files using the IceTray module `ml_suite`.

    This class sets up an IceTray pipeline that:
    - Loads input i3 files (including the GCD file),
    - Labels Monte Carlo events,
    - Runs the `ml_suite` feature extraction module,
    - Outputs results to an HDF5 file with relevant classification and extracted data.
    """

    if ClassificationConverter is not None:
        cls_converter = ClassificationConverter()
    else:
        cls_converter = None

    def extract(self) -> Path:
        """
        Executes the IceTray feature extraction pipeline on the input directory.

        Returns:
            Path: Path to the generated HDF5 output file.
        """
        Console.out(f"Running feature extraction: {self.input_dir}")
        Console.spinner().start()

        tray = I3Tray()

        # Read the i3 files to memory
        input_files = [str(self._config.pass_2_gcd_path)] + [
            str(p) for p in sorted(self.input_dir.glob("*.i3.zst"))
        ]
        tray.Add('I3Reader', Filenamelist=input_files)

        # This module labels MC events based on their topology
        tray.Add(
            MCLabeler,
            event_properties_name=None,
            mctree_name=self._config.user_config.frame_keys.mctree,
            weight_dict_name=self._config.user_config.frame_keys.weight_dict,
            bg_mctree_name=self._config.user_config.frame_keys.bg_mctree
        )

        # TODO: Debug this. Not sure why it isn't working
        # This module adds additional labels for numu events
        # tray.Add(
        #     MuonLabels,
        #     event_properties_name=None,
        #     mctree_name=self._config.user_config.frame_keys.mctree,
        #     weight_dict_name=self._config.user_config.frame_keys.weight_dict
        # )

        # This module performs the feature calculation
        tray.Add(
            ml_suite.EventFeatureExtractorModule,
            cfg_file=str(self._config.ml_suite_config_file)
        )

        # Path to output file
        outfile = self.output_dir / 'data.hdf5'

        # Serialize labels and features to HDF5
        tray.AddSegment(
            hdfwriter.I3HDFWriter,
            Output=str(outfile),
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