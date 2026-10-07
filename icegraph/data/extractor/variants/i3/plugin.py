 # Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from functools import partial
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
from .modules import is_sub_event_stream, DropCounter, event_selector, bundle_multiplicity, rebuild_mctree

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
        with suppress_output():
            from icecube.icetray import I3Tray  # pyright: ignore[reportMissingImports]
            from icecube import hdfwriter, ml_suite  # pyright: ignore[reportMissingImports]

            # required for I3CorsikaInfo
            from icecube import simclasses  # noqa: F401  # pyright: ignore[reportMissingImports]

        files = [str(self.config.gcd_path), str(item)]

        with tempfile.NamedTemporaryFile(dir=self._ctx.scratch) as out:
            tray = I3Tray()

            tray.Add("I3Reader", Filenamelist=files)

            # selection, add first so nothing downstream ever processes a dropped event
            drop_counter: DropCounter | None = None
            if self.config.selection is not None:
                drop_counter = DropCounter.from_config(self.config.selection)
                tray.Add(event_selector(self.config.selection, drop_counter, self.config.sub_event_stream))

            # rebuild the propagated tree where it was discarded, after the selection so dropped events are skipped
            if self.config.rebuild_missing_mctree is not None:
                rebuild_mctree(tray, self.config.rebuild_missing_mctree)

            # bundle multiplicity
            if self.config.multiplicity is not None:
                tray.Add(bundle_multiplicity(self.config.multiplicity))

            # mc labeler
            if self.config.mclabeler is not None:
                with suppress_output():
                    from icecube.sim_services.label_events import MCLabeler  # pyright: ignore[reportMissingImports]

                tray.Add(MCLabeler, **self.config.mclabeler)

            # ml suite
            tray.Add(
                ml_suite.EventFeatureExtractorModule,
                cfg_file=self.config.ml_suite,
                output_key="features",
                # want to only process InIceSplit frames
                If=partial(is_sub_event_stream, sub_event_stream=self.config.sub_event_stream)
            )

            tray.AddSegment(
                hdfwriter.I3HDFWriter,
                Output=out.name,
                Keys=self.config.include,
                SubEventStreams=[self.config.sub_event_stream],
                CompressionLevel=0
            )

            # suppress output from icetray if desired
            ctx = suppress_output if self.config.suppress_icetray_output else nullcontext
            with ctx():
                tray.Execute()

            # nothing is left to extract if the selection dropped every event, so break out with warning
            if drop_counter is not None and drop_counter.kept == 0 and drop_counter.dropped_events > 0:
                logger.warning(f"skipping file {item}, selection dropped all of its events: {drop_counter.dropped}")
                return None

            # load each key into a dict to save to an arrow IPC
            tables: dict[str, pl.DataFrame] = {}

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
                        {n: self._to_native(rec[n]) for n in rec.dtype.names}
                    )

        # persistent quiver dir inside scratch
        # cleaned up when the pipeline tears down scratch
        quiver_dir = Path(tempfile.mkdtemp(dir=self._ctx.scratch, prefix="quiver-"))

        # create the envelope
        env = Envelope(quiver=QuiverIPC.from_data(data=tables, root=quiver_dir))

        # register metadata
        env.set_local_attr("origin", str(item))
        env.set_global_attr("gcd", str(self.config.gcd_path))

        # settings are global so shards selected differently never load together
        # counts are per file
        if self.config.selection is not None and drop_counter is not None:
            env.set_local_attr("dropped", drop_counter.dropped)
            env.set_global_attr("selection", self.config.selection.model_dump())

        if self.config.multiplicity is not None:
            env.set_global_attr("multiplicity", self.config.multiplicity.model_dump())

        # register state
        env.state["extractor"]["src_file_ext"] = type(self).file_ext

        return env

    @staticmethod
    def _to_native(a: np.ndarray) -> np.ndarray:
        # arrow has some quirks with endianness, so need to manually check and convert if necessary
        # this will essentially never actually be a problem (most everything writes in little-endian anyway),
        # but the error is extremely cryptic and hard to diagnose so better safe than sorry

        # for subarray fields the byte order lives on the base dtype
        base = a.dtype.base
        if base.names is None and not base.isnative:
            a = a.astype(a.dtype.newbyteorder("="))
        return np.ascontiguousarray(a)

