# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from functools import cached_property, partial
from typing import ClassVar, Any, Callable
from collections.abc import Mapping

import polars as pl
import simweights
from simweights import Weighter

from icegraph.data.processor import Processor
from icegraph.data.envelope import Envelope
from icegraph.utils.hashutils import CBORBlake2B

from .config import SimWeightConfig

__all__ = ["SimWeighter"]

import logging
logger = logging.getLogger(__name__)


# bytes of the simulation name digest kept as the source code
# this fits into a float64, and is long enough to avoid collisions
_CODE_BYTES: int = 6

# distributions we know how to serialize
_DISTS: dict[str, tuple[str, ...]] = {
    "Column":                    (),
    "PowerLaw":                  ("g", "a", "b"),
    "CircleInjector":            ("radius", "cos_zen_min", "cos_zen_max"),
    "NaturalRateCylinder":       ("length", "radius", "cos_zen_min", "cos_zen_max"),
    "UniformSolidAngleCylinder": ("length", "radius", "cos_zen_min", "cos_zen_max"),
}


class SimWeighter(Processor[SimWeightConfig]):
    """Extract the per-event generation quantities and generation surface used to weight simulation."""
    name: ClassVar[str] = "i3-simweight"
    version: ClassVar[int] = 1

    def build(self) -> None:
        pass

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> SimWeightConfig:
        return SimWeightConfig(**config)

    @cached_property
    def _weighter_constructor(self) -> Callable[[Any], Weighter]:
        match self.config.simulation:
            case "nugen":
                return partial(simweights.NuGenWeighter, nfiles=1)
            case "corsika":
                return partial(simweights.CorsikaWeighter, nfiles=1)

        raise KeyError(f"No supported weighter found for key {self.config.simulation!r}.")

    @cached_property
    def _sim_code(self) -> int:
        digest = CBORBlake2B()(self.config.simulation)

        # truncate to fit as float64
        # this should be plenty such that no sim codes will overlap
        return int(digest[:2 * _CODE_BYTES], 16)

    def _resolve_ids(
            self, frames: Mapping[str, pl.DataFrame], ids: list[str], height: int
    ) -> pl.DataFrame:
        """Take the ids keying the weight columns from the weighting tables themselves."""
        reference: pl.DataFrame | None = None
        source: str | None = None
        rejected: dict[str, str] = {}

        for key in frames:
            frame = frames[key]

            # if more id cols were identified than cols in the table, then table cannot hold ids
            if set(ids) > set(frame.columns):
                rejected[key] = "missing id columns"
                continue

            # frame needs same number of entries
            if frame.height != height:
                rejected[key] = f"{frame.height} rows"
                continue

            frame = frame.select(ids)

            # a table repeating an id is indexed by something finer than the event
            # so it cannot be the source of a per-event weight
            if frame.is_duplicated().any():
                rejected[key] = "ids are not unique"
                continue

            # verify all tables that hold id cols agree
            if reference is None:
                reference, source = frame, key
                continue

            if not frame.equals(reference):
                raise RuntimeError(
                    f"{type(self).name}: tables {source!r} and {key!r} both look like the source "
                    f"of the weight columns but disagree on the id columns {ids}. Cannot determine "
                    f"which rows the weights belong to."
                )

        if reference is None:
            raise RuntimeError(
                f"{type(self).name}: no weighting table carries the id columns {ids} over "
                f"{height} unique rows, so the weight columns cannot be keyed. Tables "
                f"considered: {rejected}."
            )

        logger.debug("stage=%s: keyed %d generation rows on %s", type(self).name, height, source)

        return reference

    def _serialize_dist(self, dist: Any) -> dict[str, Any]:
        """Describe one distribution as the parameters needed to reconstruct it."""
        kind = type(dist).__name__
        names = _DISTS.get(kind)

        # an unknown distribution would rebuild silently wrong, so refuse to write it
        if names is None:
            raise TypeError(
                f"{type(self).__name__}: cannot serialize distribution {kind!r} ({dist!r}). "
                f"Known: {sorted(_DISTS)}."
            )

        (colname,) = dist.columns
        if colname is None:
            raise ValueError(
                f"{type(self).__name__}: distribution {kind!r} carries no column name, so the "
                f"rebuilt surface could not be weighted."
            )

        params = {name: float(getattr(dist, name)) for name in names}
        params["colname"] = colname

        return {"kind": kind, "params": params}

    def _serialize_surface(self, weighter: Weighter) -> dict[str, Any]:
        """Describe the generation surface as the parameters needed to rebuild it on load."""
        components: dict[str, Any] = {}

        # spectra is keyed by pdgid, each key holding the components generated for it.
        # flattened here since every component names its own pdgid
        for spectra in weighter.surface.spectra.values():
            for spec in spectra:
                components[str(len(components))] = {
                    "pdgid": int(spec.pdgid),
                    "nevents": float(spec.nevents),

                    # ordered, simweights compares dists sequence-wise when merging surfaces
                    "dists": {str(i): self._serialize_dist(d) for i, d in enumerate(spec.dists)}
                }

        return {
            "__simweights_version__": simweights.__version__,
            "simulation": self.config.simulation,

            # what the source column holds and where
            "code": self._sim_code,
            "column": "sim_code",

            "data": components
        }

    def _process(self, item: Envelope) -> Envelope | None:
        # only the tables the weighter reads, so nothing unrelated is offered to it
        # or considered when the weight columns are keyed
        tables = item.resolve_cols(self.config.tables)
        quiver = item.quiver.subset(tables)

        # build the weighter
        weighter = self._weighter_constructor(quiver)

        # resolve all cols
        columns = [pl.Series(name, weighter.get_weight_column(name)) for name in weighter.colnames]

        # resolve ids and build id only DF
        ids = item.resolve_cols(self.config.ids)
        frame = self._resolve_ids(quiver, ids, columns[0].len())

        # build sim code column
        columns += [pl.lit(self._sim_code, dtype=pl.Int64).alias("sim_code")]

        # add data and register to envelope
        item.tmp[self.config.to] = frame.with_columns(columns)

        # one surface per file, summed across shards on load
        item.set_local_attr(self.config.attr, self._serialize_surface(weighter))

        return item
