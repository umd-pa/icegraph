# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import ClassVar, Any
import tempfile
from pathlib import Path

import pandas as pd
import h5py

from icegraph.utils.stdout import suppress_output
from icegraph.data.envelope import Envelope
from icegraph.data.extractor import Extractor

from .config import I3ExtractorConfig

with suppress_output():
    from icecube.icetray import I3Tray          # pyright: ignore[reportMissingImports]
    from icecube import hdfwriter, ml_suite     # pyright: ignore[reportMissingImports]

__all__ = ["I3Extractor"]

import logging
logger = logging.getLogger(__name__)


class I3Extractor(Extractor[I3ExtractorConfig]):
    """Extracts features from I3 files using the IceTray module `ml_suite`."""
    name: ClassVar[str] = "i3"
    version: ClassVar[int] = 1

    file_ext: ClassVar[str] = "i3.zst"

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> I3ExtractorConfig:
        return I3ExtractorConfig(**config)

    def build(self) -> None:
        return

    def _process(self, item: Path) -> Envelope | None:
        files = [str(self.config.gcd_path), str(item)]

        with tempfile.NamedTemporaryFile() as out:
            tray = I3Tray()

            tray.Add("I3Reader", Filenamelist=files)
            tray.Add(ml_suite.EventFeatureExtractorModule, cfg_file=self.config.ml_suite, output_key="features")

            tray.AddSegment(
                hdfwriter.I3HDFWriter,
                Output=out.name,
                Keys=self.config.include,
                SubEventStreams=["InIceSplit"]
            )

            # suppress garbage output from icetray
            with suppress_output():
                tray.Execute()

            # load each into a dict to pass to envelope
            data: dict[str, pd.DataFrame] = {}
            with h5py.File(out.name, "r") as f:

                # ensure key exists in file
                available = list(f.keys())
                for key in self.config.include:
                    if key not in available:
                        # if skip missing is set to True, just skip the file and continue
                        if self.config.skip_missing:
                            logger.warning(f"skipping file {item}, missing key '{key}', available keys: {available}")
                            return None

                        # if skip missing is set to False, raise and break out
                        raise KeyError(
                            f"Missing HDF5 key '{key}' for input file {item}. Available keys: {available}"
                        )

                    dset = f[key]
                    assert isinstance(dset, h5py.Dataset)  # narrow type union at runtime
                    data[key] = pd.DataFrame(dset[:])

        # create the envelope
        env = Envelope(data=data)

        # register metadata
        env.set_local_attr("origin", str(item))
        env.set_global_attr("gcd", str(self.config.gcd_path))

        # register state
        env.state["extractor"]["src_file_ext"] = type(self).file_ext

        return env

