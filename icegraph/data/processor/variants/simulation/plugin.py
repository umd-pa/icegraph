# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import ClassVar, Any

import numpy as np
import polars as pl
import simweights

from icegraph.data.processor import Processor
from icegraph.data.envelope import Envelope
from icegraph.utils.hashutils import CBORBlake2B

from .config import I3SimulationConfig

__all__ = ["I3Simulation"]

import logging
logger = logging.getLogger(__name__)


# bytes of the settings digest kept as the sim code
# this fits into a float64, and is long enough to avoid collisions
_CODE_BYTES: int = 6

# generation column each record's sim code is written to
_COLUMN: str = "sim_code"

# nugen generation shared by every type a file threw, so the same whichever it kept
_NUGEN_SETTINGS: tuple[str, ...] = (
    "NEvents", "MinEnergyLog", "MaxEnergyLog", "PowerLawIndex", "MinZenith", "MaxZenith",
    "InjectionSurfaceR", "CylinderHeight", "CylinderRadius"
)

# the generation columns are pinned to these dtypes rather than kept as the tables
# happen to type them. the compress stage records the source dtype of every packed
# column, the set id hashes it, and the read side casts each column back with it, so a
# table that types a column differently gives its shards a set id that cannot be loaded
# alongside the rest. corsika carries pdgid as a double and no per event weight, nugen
# carries an int32 pdgid and a float weight, but they must have the same dtype
_GEN_DTYPE_DEFAULT: type[pl.DataType] = pl.Float64
_GEN_DTYPES: dict[str, type[pl.DataType]] = {
    "pdgid": pl.Int32
}


def _polars_to_numpy(frame: pl.DataFrame) -> dict[str, np.ndarray]:
    return {name: frame.get_column(name).to_numpy() for name in frame.columns}


class I3Simulation(Processor[I3SimulationConfig]):
    """Store the simulation a file was generated with, and extract the per-event quantities used to weight it."""
    name: ClassVar[str] = "i3-simulation"
    version: ClassVar[int] = 1

    def build(self) -> None:
        return

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> I3SimulationConfig:
        return I3SimulationConfig(**config)

    def _sim_code(self, events: pl.DataFrame) -> int:
        """A code every file of a set shares whatever it kept, and sets thrown differently do not."""
        # construct a hash over values that identify a specific sim source
        settings: dict[str, Any] = {"type": self.config.simulation}

        if self.config.simulation == "nugen":
            # get all flavors in the file
            flavors = np.unique(np.abs(events.get_column("PrimaryNeutrinoType").to_numpy()))

            # if more than one flavor, raise and request manual sim_code
            if len(flavors) != 1:
                raise RuntimeError(
                    f"{type(self).name}: the block holds neutrinos of flavors {flavors.tolist()}, so "
                    f"its set cannot be told from its generation. Set 'sim_code' to name it."
                )

            # include flavor to hash
            # on load a different surface is built per flavor,
            # these surfaces are then summed before building the weighter
            # all nugen events regardless of flavor are then routed to the combined nugen weighter
            # sim code needs to be different per flavor so the weighter can tell which files
            # simulated which flavor to count them correctly
            settings["flavor"] = float(flavors[0])

            # include all nugen settings in the hash, files with a different energy range
            # (or zenith range, cylinder radius, etc) need to be assigned
            # separate sim codes like with flavors to get correct file counts
            # these are summed into a merged surface before constructing the weighter
            settings |= {name: events.get_column(name)[0] for name in _NUGEN_SETTINGS if name in events.columns}

        # truncate to fit as float64
        return int(CBORBlake2B()(settings)[:2 * _CODE_BYTES], 16)

    def _check_nugen_set(self, events: pl.DataFrame) -> None:
        """Raise unless every file of the block threw the same generation."""
        # only one row per type is stored, so the files of a block must agree on these
        varying = [
            name for name in _NUGEN_SETTINGS if name in events.columns and events.get_column(name).n_unique() > 1
        ]

        # the share of each particle type must be the same in every file of a set
        if "TypeWeight" in events.columns:
            shares = events.group_by("PrimaryNeutrinoType").agg(pl.col("TypeWeight").n_unique())
            if (shares.get_column("TypeWeight") > 1).any():
                varying.append("TypeWeight")

        if varying:
            raise RuntimeError(
                f"{type(self).name}: the files of a block differ in {varying}, so they were thrown by different "
                f"sets. Keep each set in its own directory and collect with 'group_by: parent', or process seperately."
            )

    def _process(self, item: Envelope) -> Envelope | None:
        quiver, config = item.quiver, self.config

        # the collector records every input file of the block, with events or without
        sources = item.get_local_attr("sources")
        if not sources:
            raise RuntimeError(f"{type(self).name}: no source files found in the envelope.")

        # simweights reads the per event quantities off the tables under its own names
        weight_dict: pl.DataFrame | None = None
        match config.simulation:
            case "nugen":
                # load eventwise generation info and nugen sim info table
                # (nugen sim info is duplicated per event alongside generation info for some reason)
                weight_dict = quiver[config.weight_dict]
                self._check_nugen_set(weight_dict)

                # construct the weighter
                weighter = simweights.NuGenWeighter({"I3MCWeightDict": _polars_to_numpy(weight_dict)}, nfiles=1)

                # stash generation info
                # store one row per type
                generation = {
                    "weight_dict": _polars_to_numpy(
                        weight_dict.unique(subset="PrimaryNeutrinoType", keep="first", maintain_order=True))
                }

            case "corsika":
                # ensure every file contains an S frame (required)
                missing = [source for source, part in zip(sources, quiver.parts) if config.corsika_info not in part]
                if missing:
                    raise RuntimeError(f"{type(self).name}: the files {missing} hold no '{config.corsika_info}'.")

                # load eventwise generation info and corsika sim info tables
                weight_dict = quiver[config.primary]
                info = _polars_to_numpy(quiver[config.corsika_info])

                # construct the weighter
                weighter = simweights.CorsikaWeighter({"I3CorsikaInfo": info, "PolyplopiaPrimary": _polars_to_numpy(
                    weight_dict)})

                # stash generation info
                generation = {"corsika_info": info}

            case _:  # pragma: no cover
                # cannot technically be reached as pydantic validates this on init, including to be explicit
                raise ValueError(
                    f"{type(self).name}: unknown simulation {config.simulation!r}"
                )

        # weight_dict is assigned by now
        assert weight_dict is not None

        # compute the sim code if not given
        code = self._sim_code(weight_dict) if config.sim_code is None else config.sim_code

        # resolve all cols, each pinned to the dtype every simulation type must agree on
        columns = [
            pl.Series(name, weighter.get_weight_column(name))
            .cast(_GEN_DTYPES.get(name, _GEN_DTYPE_DEFAULT))
            for name in weighter.colnames
        ]

        # the weight columns are read off the event table, so its ids key them
        ids = weight_dict.select(item.resolve_cols(config.ids))
        if ids.is_duplicated().any():
            raise RuntimeError(
                f"{type(self).name}: the ids {ids.columns} repeat in the event table, so the weight "
                f"columns cannot be keyed."
            )

        # add a sim code column which routes each record to the correct surface/weighter on load
        columns += [pl.lit(code, dtype=pl.Int64).alias(_COLUMN)]

        item.tmp[config.to] = ids.with_columns(columns)
        item.set_local_attr("simulation", {
            "sim_code": code,
            "type": config.simulation,
            "column": _COLUMN,

            # files the generation was thrown over, counting those without events
            "nfiles": len(sources),
            **generation
        })

        return item
