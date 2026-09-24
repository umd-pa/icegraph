# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterator, Literal

import numpy as np

from icegraph.common.data import AttributeDomain
from icegraph.common.record import Attributes, validate_attrs

from .schema import SimulationAttrs

__all__ = ["SurfaceSet", "build_surfaces"]

import logging
logger = logging.getLogger(__name__)


# local attribute the i3-simulation processor writes, absent for real data
_KEY: str = "simulation"

_TABLES: dict[Literal['nugen', 'corsika'], str] = {
    "corsika": "corsika_info",
    "nugen": "weight_dict"
}

# simweights builds a surface through a weighter, which also reads the event columns
# only the surface is used so no events are offered
_NO_EVENTS: dict[str, np.ndarray] = {name: np.empty(0) for name in ("type", "energy", "zenith")}


@dataclass(frozen=True)
class SurfaceSet:
    """The generation surfaces of a dataset, and how a record is routed to one."""
    # simulation type keyed to the surface its sets were generated on
    surfaces:   dict[str, Any]
    codes:      dict[int, str]

    # generation column carrying the sim code
    column:     str


def build_surfaces(attrs: Callable[[], Iterator[Attributes]]) -> SurfaceSet | None:
    """Build the generation surface of every simulation type the shards carry, None for real data."""
    import simweights

    # sim code keyed to its type and the table its generation was stored as
    codes: dict[int, str] = {}
    tables: dict[int, list[dict[str, np.ndarray]]] = {}
    columns: set[str] = set()
    real = 0

    for attr in attrs():
        raw = attr[AttributeDomain.LOCAL].get(_KEY)

        if raw is None:
            real += 1
            continue

        # pydantic for attr validation
        sim_attrs = validate_attrs(SimulationAttrs, raw, source=f"shard ID={attr.shard_id}")

        # extract base values
        sim = sim_attrs.simulation
        code = sim_attrs.sim_code
        column = sim_attrs.column

        # stash the sim_code column name declared by this shard
        columns.add(column)

        # a code is one set so ensure its files agree on how it was stored
        owner = codes.setdefault(code, sim)
        if owner != sim:
            raise RuntimeError(
                f"Shard ID={attr.shard_id} stores sim code {code} as {sim} "
                f", which another shard stores as {owner}."
            )

        # ensure the correct generation table info was stored for the declared sim source
        generation_table_name = _TABLES.get(sim)
        if generation_table_name is None:
            raise ValueError(f"Shard ID={attr.shard_id}: unknown simulation type {sim!r}.")

        # load the generation table, raising if not present
        generation_table = getattr(sim_attrs, generation_table_name)
        if generation_table is None:
            raise ValueError(
                f"Shard ID={attr.shard_id} declares its source simulation as {sim!r}. Expected "
                f"a corresponding generation table {generation_table_name!r}, but it was not found."
            )

        # store lists of all gen tables keyed by sim_code
        tables.setdefault(code, []).append(generation_table)

    # if codes is empty, no shards were simulation and can safely return None
    if not codes:
        return None

    # if some shards declared simulation, and real data is mixed in, raise as this is not allwoed
    if real:
        raise RuntimeError(
            f"{real} shard(s) carry no simulation while others do. Real data and simulation "
            f"cannot be weighted together."
        )

    # ensure all shards agree on sim_code column declaration
    if len(columns) != 1:
        raise RuntimeError(
            f"Shards disagree on which column carries the sim code ({sorted(columns)})."
        )

    # construct surfaces
    surfaces: dict[str, Any] = {}
    for code in sorted(codes):
        sim = codes[code]

        merged = {col: np.concatenate([table[col] for table in tables[code]]) for col in tables[code][0]}

        # construct based on sim source
        match sim:
            case "nugen":
                surface = simweights.NuGenWeighter({"I3MCWeightDict": merged}, nfiles=len(tables[code])).surface
            case "corsika":
                surface = simweights.CorsikaWeighter({"I3CorsikaInfo": merged, "PolyplopiaPrimary": _NO_EVENTS}).surface
            case _:
                raise ValueError(f"build_surfaces: unknown simulation {sim!r}.")

        surfaces[sim] = surfaces[sim] + surface if sim in surfaces else surface

        logger.info(
            "[StandardI3Decoder] Built the generation surface for %s (sim code %d) over %d loaded file(s).",
            sim, code, len(tables[code])
        )

    return SurfaceSet(
        surfaces=surfaces,
        codes=codes,
        column=columns.pop()
    )
