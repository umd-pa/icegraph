# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import ClassVar, Any
from functools import cached_property

import numpy as np

from icecube import dataio                  # pyright: ignore[reportMissingImports]
from icecube.icetray import OMKey           # pyright: ignore[reportMissingImports]
from icecube.dataclasses import I3Geometry  # pyright: ignore[reportMissingImports]

from icegraph.data.processor import Processor
from icegraph.data.envelope import Envelope

from .config import DOMConfig

__all__ = ["DOMProcessor"]


class DOMProcessor(Processor[DOMConfig]):
    """Processor to convert DOM IDs to cartesian position."""
    name: ClassVar[str] = "domproc"
    version: ClassVar[int] = 1

    _geometry: I3Geometry | None

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> DOMConfig:
        return DOMConfig(**config)

    def build(self) -> None:
        self._geometry = None

    def _process(self, item: Envelope) -> Envelope | None:
        active = self._require_active(item)
        main = item.tmp[active]

        # get geometry frame from gcd if not yet stashed
        if self._geometry is None:
            gcd_path = item.get_global_attr("gcd", None)

            # ensure a gcd path was provided
            if gcd_path is None:
                raise RuntimeError("GCD file path must be provided to compute DOM coordinates.")

            self._geometry = self._get_geometry(gcd_path)

        # get string, om, pmt cols
        cols = item.resolve_cols(self.config.cols)
        out = item.resolve_cols(self.config.out)

        # get list of all unique dom ids in frame
        lut = main[cols].drop_duplicates(subset=cols).copy()

        # compute positions from ids
        dom_ids = lut.to_numpy(dtype=np.int64, copy=False)
        pos = np.empty((len(lut), 3), dtype=np.float64)  # allocate empty array
        for i, (s, om, pmt) in enumerate(dom_ids):
            pos[i] = self._id_to_position(int(s), int(om), int(pmt))

        # add back to lut as new columns
        lut[out] = pos

        # merge to tmp and return
        return item.merge(lut, to=active, on=cols, validate="m:1")

    @cached_property
    def geometry(self) -> I3Geometry:
        if self._geometry is None:
            raise RuntimeError("Cannot get geometry frame before first file has been passed through.")
        return self._geometry

    def _id_to_position(self, string: int, om: int, pmt: int) -> tuple[float, float, float]:
        # get position and return as tuple
        p = self.geometry.omgeo[OMKey(string, om, pmt)].position
        return p.x, p.y, p.z

    @staticmethod
    def _get_geometry(path: str) -> I3Geometry:
        for frame in dataio.I3File(path):
            if "I3Geometry" in frame:
                return frame["I3Geometry"]

        # if geometry was not found, raise an exception
        raise RuntimeError(f"Could not find geometry frame in GCD file: {path}")
