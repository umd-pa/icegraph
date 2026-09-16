# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from typing import Any, Callable, Iterator

from icegraph.common.data import AttributeDomain
from icegraph.common.record import Attributes, validate_attrs

from .schema import Dist, SurfaceAttrs, SurfaceComponent, dist_args, in_order

__all__ = ["SurfaceSet", "build_surfaces"]

import logging
logger = logging.getLogger(__name__)


# the only release this rebuild targets
# patchwork until the surface serializes itself upstream in next release
_SUPPORTED_VERSION: str = "0.1.3"

# the package root exports every distribution but this one, so only its module is
# pinned, and only it moves if the internals are rearranged
_DIST_ROOT: str = "simweights"
_DIST_MODULES: dict[str, str] = {
    "Column": "simweights._utils",
}


@dataclass(frozen=True)
class SurfaceSet:
    """The generation surfaces of a dataset, and how a record is routed to one."""
    # simulation type keyed to the surface its shards were generated on
    surfaces:   dict[str, Any]

    # source code keyed to the simulation type it stands for
    codes:      dict[int, str]

    # generation column carrying the sim code
    column:     str


def _build_component(component: SurfaceComponent) -> Any:
    """Rebuild the surface of one generated species."""
    import simweights

    # iterate over each distribution in order
    dists: list[Dist] = []
    ordered_dist_structs = [component.dists[key] for key in sorted(component.dists)]
    for dist_struct in ordered_dist_structs:
        # load the module where this distribution class lives in simweights
        module = _DIST_MODULES.get(dist_struct.kind, _DIST_ROOT)

        # load the class object from that module
        try:
            cls = getattr(importlib.import_module(module), dist_struct.kind)
        except (ImportError, AttributeError) as e:
            raise TypeError(
                f"Could not resolve {dist_struct.kind!r} from module {module!r}"
            ) from e

        # extract params as dict from pydantic
        params = dist_struct.params.model_dump()

        # instantiate and append
        instance = cls(**params)
        dists.append(instance)

    factory = getattr(simweights, "generation_surface", None)
    if factory is None:
        raise AttributeError(
            f"simweights {simweights.__version__} exposes no generation_surface()"
        )

    # multiply by nevents to normalize
    return factory(component.pdgid, *dists) * component.nevents


def build_surfaces(
        attrs: Callable[[], Iterator[Attributes]], *, key: str
) -> SurfaceSet:
    """Sum the per-file generation surfaces of every shard, by simulation type.

    Each shard carries the surface of the file it came from, generated with
    ``nfiles=1``, so the sum over the shards of one simulation type is the surface
    that type was generated on. Types are summed apart rather than together: a
    dataset may mix them, and each is weighted against its own flux.

    Every shard is expected to carry a surface.
    """
    import simweights

    # can only support 0.1.3 for now since rebuild is manual with internal objects
    if simweights.__version__ != _SUPPORTED_VERSION:
        raise RuntimeError(
            f"The generation surface rebuild targets simweights {_SUPPORTED_VERSION}, but "
            f"{simweights.__version__} is installed."
        )

    # dict of surfaces built from all shards
    surfaces: dict[str, Any] = {}

    # track event count covered by shards to ensure matches loaded dataset size
    nevents: dict[str, float] = {}

    # ensure simulations are one to one with codes
    codes: dict[int, str] = {}

    # make sure code column matches across shards
    columns: set[str] = set()

    for attr in attrs():
        raw = attr[AttributeDomain.LOCAL].get(key)

        # ensure root is present
        if raw is None:
            raise KeyError(
                f"Missing key {key!r} in shard local attributes (shard ID={attr.shard_id})."
            )

        blob = validate_attrs(SurfaceAttrs, raw, source=f"shard ID={attr.shard_id}")

        if blob.version != _SUPPORTED_VERSION:
            raise RuntimeError(
                f"Found generation data produced by simweights {blob.version}, "
                f"which is different from simweights {_SUPPORTED_VERSION} (shard ID = {attr.shard_id})."
            )

        columns.add(blob.column)

        # the shard says what its records carry, so no code is ever assumed here
        owner = codes.setdefault(blob.code, blob.simulation)
        if owner != blob.simulation:
            raise RuntimeError(
                f"Shard ID={attr.shard_id} labels its {blob.simulation} generation with source code "
                f"{blob.code}, which another shard already uses for {owner}. The two are indistinguishable, "
                f"so they cannot be weighted together."
            )

        # order doesnt matter for components, iterate over values, keys are just ordering ints
        for component in blob.data.values():

            # if no tracked count for simulation, make one and set to 0
            if nevents.get(blob.simulation) is None:
                nevents[blob.simulation] = 0

            # add component nevents to tally to track how many events the surfaces cover
            nevents[blob.simulation] += component.nevents

            # rebuild from component
            rebuilt = _build_component(component)

            if surfaces.get(blob.simulation) is None:
                # no surface yet for this simulation
                surfaces[blob.simulation] = rebuilt

            else:
                # surface already exists, add to it
                surfaces[blob.simulation] += rebuilt

    if not surfaces:
        raise RuntimeError(
            "No generation surface was found in any shard."
        )

    if len(columns) != 1:
        raise RuntimeError(
            f"Shards disagree on which column carries the source code ({sorted(columns)})."
        )

    for simulation, surface in surfaces.items():
        # a parameter this format does not carry would rebuild silently wrong, and the
        # event count is the one number that catches a changed surface scaling convention
        total = sum(
            spec.nevents
            for spectra in surface.spectra.values()
            for spec in spectra
        )
        want = nevents[simulation]

        if not math.isclose(total, want, rel_tol=1e-9):
            raise RuntimeError(
                f"The rebuilt {simulation} generation surface accounts for {total:g} events but "
                f"the shards recorded {want:g}."
            )

        logger.info(
            "[StandardI3Decoder] Rebuilt generation surfaces over all loaded shards for simulation '%s' (%g events).",
            simulation, total
        )

    # columns is enforced to have len 1, so just take that value
    column = columns.pop()
    return SurfaceSet(surfaces=surfaces, codes=codes, column=column)
