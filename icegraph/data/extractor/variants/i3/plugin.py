# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import ClassVar, Any
import tempfile
from pathlib import Path
from contextlib import nullcontext

import numpy as np
import polars as pl
import h5py

from icegraph.utils.stdout import suppress_output
from icegraph.data.envelope import Envelope
from icegraph.data.extractor import Extractor
from icegraph.data.quiver import QuiverIPC

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

        with tempfile.NamedTemporaryFile(dir=self._ctx.scratch) as out:
            tray = I3Tray()

            tray.Add("I3Reader", Filenamelist=files)
            tray.Add(ml_suite.EventFeatureExtractorModule, cfg_file=self.config.ml_suite, output_key="features")

            tray.AddSegment(
                hdfwriter.I3HDFWriter,
                Output=out.name,
                Keys=self.config.include,
                SubEventStreams=["InIceSplit"],
                CompressionLevel=0
            )

            # suppress output from icetray if desired
            ctx = suppress_output if self.config.suppress_icetray_output else nullcontext
            with ctx():
                tray.Execute()

            # load each key into a dict to save to an arrow IPC
            tables: dict[str, pl.DataFrame] = {}

            # persistent quiver dir inside scratch
            # cleaned up when the pipeline tears down scratch
            quiver_dir = Path(tempfile.mkdtemp(dir=self._ctx.scratch, prefix="quiver-"))

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
                            f"Missing key '{key}' for input file {item}. Available keys: {available}"
                        )

                    dset = f[key]
                    assert isinstance(dset, h5py.Dataset)  # narrow type union at runtime
                    rec = dset[:]
                    tables[key] = pl.DataFrame(
                        {n: np.ascontiguousarray(rec[n]) for n in rec.dtype.names}
                    )

        # create the envelope
        env = Envelope(quiver=QuiverIPC.from_data(data=tables, root=quiver_dir))

        # register metadata
        env.set_local_attr("origin", str(item))
        env.set_global_attr("gcd", str(self.config.gcd_path))

        # register state
        env.state["extractor"]["src_file_ext"] = type(self).file_ext

        return env

