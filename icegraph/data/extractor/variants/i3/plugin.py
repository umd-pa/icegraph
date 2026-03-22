# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import ClassVar, Any
import tempfile
from pathlib import Path

import pandas as pd
import h5py

from icegraph.utils.stdout import suppress_output
from icegraph.exceptions import IceCubeImportError
from icegraph.data.types import Envelope
from icegraph.data.extractor import Extractor
from icegraph.types.data import AttributeDomain

from .config import I3ExtractorConfig

with suppress_output():
    try:
        from icecube.icetray import I3Tray
        from icecube import hdfwriter, ml_suite
        from icecube.sim_services.label_events import MCLabeler
    except ImportError:
        I3Tray = IceCubeImportError.IceCubeMissingBase  # type: ignore[assignment]
        MCLabeler = IceCubeImportError.IceCubeMissingBase  # type: ignore[assignment]
        hdfwriter = IceCubeImportError()  # type: ignore[assignment]
        ml_suite = IceCubeImportError()  # type: ignore[assignment]

__all__ = ["I3Extractor"]


class I3Extractor(Extractor[I3ExtractorConfig]):
    """Extracts features from I3 files using the IceTray module `ml_suite`."""
    name: ClassVar[str] = "i3_extractor"
    version: ClassVar[int] = 1

    file_ext: ClassVar[str] = "i3.zst"

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> I3ExtractorConfig:
        return I3ExtractorConfig(**config)

    def build(self) -> None:
        return

    def _process(self, infile: Path) -> Envelope | None:
        files = [str(self.config.gcd_path), str(infile)]

        with tempfile.NamedTemporaryFile() as out:
            tray = I3Tray()

            tray.Add("I3Reader", Filenamelist=files)
            tray.Add(ml_suite.EventFeatureExtractorModule, cfg_file=self.config.ml_suite, output_key="features")
            tray.Add(MCLabeler, **self.config.mclabeler)

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
                for key in self.config.include:
                    data[key] = pd.DataFrame(f[key][:])

        # create the envelope
        env = Envelope(data=data)

        # register metadata
        env.attrs[AttributeDomain.LOCAL.name]["origin"] = str(infile)
        env.attrs[AttributeDomain.GLOBAL.name]["gcd"] = str(self.config.gcd_path)

        # register state
        env.state["extractor"]["src_file_ext"] = type(self).file_ext

        return env

