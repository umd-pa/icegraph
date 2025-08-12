# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from pathlib import Path
from typing import Optional, Union, Sequence, List

from icegraph.console import Console
from icegraph.console.streams import suppress_output
from .base import IGExtractor
from .base.modules import UniqueID
from icegraph.pathutils import PathResolver, PathValidator
from .base.exceptions import MissingI3FilesError
from icegraph.exceptions import IceCubeImportError

# have to wrap in try/except block so sphinx can properly generate docs
try:
    from icecube.icetray import I3Tray as _I3Tray, I3Module as _I3Module
    from icecube import (
        dataclasses as _dataclasses,
        icetray as _icetray,
        dataio as _dataio,
        hdfwriter as _hdfwriter,
        ml_suite as _ml_suite
    )
    from icecube.sim_services.label_events import (
        MCLabeler as _MCLabeler,
        ClassificationConverter as _ClassificationConverter,
        MuonLabels as _MuonLabels
    )
except ImportError:
    _dataclasses = IceCubeImportError()
    _icetray = IceCubeImportError()
    _dataio = IceCubeImportError()
    _hdfwriter = IceCubeImportError()
    _ml_suite = IceCubeImportError()

    _I3Tray = IceCubeImportError.IceCubeMissingBase
    _I3Module = IceCubeImportError.IceCubeMissingBase
    _MCLabeler = IceCubeImportError.IceCubeMissingBase
    _ClassificationConverter = IceCubeImportError.IceCubeMissingBase
    _MuonLabels = IceCubeImportError.IceCubeMissingBase

dataclasses = _dataclasses
icetray = _icetray
dataio = _dataio
hdfwriter = _hdfwriter
ml_suite = _ml_suite

I3Tray = _I3Tray
I3Module = _I3Module
MCLabeler = _MCLabeler
ClassificationConverter = _ClassificationConverter
MuonLabels = _MuonLabels

__all__ = ["FeatureExtractor"]


class FeatureExtractor(IGExtractor):
    """
    Extracts features from I3 files using the IceTray module `ml_suite`.

    This class sets up an IceTray pipeline that:
    - Loads input I3 files (including the GCD file),
    - Labels Monte Carlo events,
    - Runs the `ml_suite` feature extraction module,
    - Outputs results to an HDF5 file with relevant classification and extracted data.
    """

    cls_converter = ClassificationConverter and ClassificationConverter()

    def extract(self, outfile: Optional[Union[str, Path]] = None) -> Path:
        """
        Executes the IceTray feature extraction pipeline on the input directory.

        Returns:
            Path: Path to the generated HDF5 output file.
        """
        self._file_paths: List[str] = PathResolver.normalize_sources(self._source, ".i3.zst", use_str=True)
        source_repr = Console.source_repr(self._source)

        Console.banner("Feature Extractor")
        Console.out(f"Running feature extraction: {source_repr}")

        if not self._file_paths:
            raise MissingI3FilesError(f"No I3 files found in source {source_repr}")

        # Path to output file
        resolver = PathResolver(path=outfile, origin=None, extension=None, stage="extractor")
        outdir = resolver.resolve(return_dir=True)

        for infile in Console.progress_bar(self._file_paths):
            tray = I3Tray()

            # get the i3 file paths and output hdf5 file path
            input_files = [str(self._config.gcd_path), infile]
            outfile = outdir / Path(infile).with_suffix(".hdf5").name

            tray.Add('I3Reader', Filenamelist=input_files)

            # This module labels MC events based on their topology
            # TODO: make this optional
            tray.Add(
                MCLabeler,
                event_properties_name=None,
                mctree_name=self._config.user_config.frame_keys.mctree,
                weight_dict_name=self._config.user_config.frame_keys.weight_dict,
                bg_mctree_name=self._config.user_config.frame_keys.bg_mctree
            )

            # This module performs the feature calculation
            tray.Add(
                ml_suite.EventFeatureExtractorModule,
                cfg_file=str(self._config.ml_suite_config_file)
            )

            tray.Add(UniqueID)

            # Serialize labels and features to HDF5
            tray.AddSegment(
                hdfwriter.I3HDFWriter,
                Output=str(outfile),
                Keys=[
                    "ml_suite_features",
                    ("classification", self.cls_converter),
                    "classification_emuon_entry",
                    "classification_emuon_deposited",
                    self._config.user_config.frame_keys.truth_dict
                ],
                SubEventStreams=["InIceSplit"]
            )

            with suppress_output():
                tray.Execute()

        Console.out(f"Output files saved to {outdir}")
        return outdir
