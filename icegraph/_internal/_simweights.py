# SPDX-FileCopyrightText: © 2022 the SimWeights contributors
#
# SPDX-License-Identifier: BSD-2-Clause

"""
Temporary shim reproducing icecube/simweights#87 (surface serialization).

This is more or less a direct copy of the contents of the PR.
TREAT AS UNSTABLE UNTIL MERGE
"""

from __future__ import annotations

from functools import singledispatch
from typing import TYPE_CHECKING, Any

from simweights._generation_surface import CompositeSurface, GenerationSurface
from simweights._powerlaw import PowerLaw
from simweights._spatial import (
    CircleInjector,
    CylinderBase,
    NaturalRateCylinder,
    UniformSolidAngleCylinder,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


_POWERLAW_CLASSES = {
    cls.__name__: cls
    for cls in (PowerLaw,)
}
_SPATIAL_CLASSES = {
    cls.__name__: cls
    for cls in (CylinderBase, UniformSolidAngleCylinder, NaturalRateCylinder, CircleInjector)
}


# class resolution

def resolve_powerlaw(name: str) -> type:
    """Resolve a powerlaw class object from its name."""
    if name not in _POWERLAW_CLASSES:
        raise ValueError(
            f"resolve_powerlaw: unknown power law class {name!r}, expected one of {sorted(_POWERLAW_CLASSES)}"
        )

    return _POWERLAW_CLASSES[name]


def resolve_spatial(name: str) -> type:
    """Resolve a spatial distribution class object from its name."""
    if name not in _SPATIAL_CLASSES:
        raise ValueError(
            f"resolve_spatial: unknown spatial distribution class {name!r}, expected one of {sorted(_SPATIAL_CLASSES)}"
        )

    return _SPATIAL_CLASSES[name]


# to_dict

@singledispatch
def to_dict(obj: Any) -> dict[str, Any]:
    """Serialize a simweights object to json safe state."""
    # upstream has no equivalent branch
    raise TypeError(f"to_dict: no serializer for {type(obj).__name__}")


@to_dict.register
def _(obj: PowerLaw) -> dict[str, float]:
    # json safe state
    return {param: float(getattr(obj, param)) for param in ("g", "a", "b")}


@to_dict.register
def _(obj: CylinderBase) -> dict[str, float]:
    # json safe state
    return {param: float(getattr(obj, param)) for param in ("length", "radius", "cos_zen_min", "cos_zen_max")}


@to_dict.register
def _(obj: CircleInjector) -> dict[str, float]:
    # json safe state
    return {param: float(getattr(obj, param)) for param in ("radius", "cos_zen_min", "cos_zen_max")}


@to_dict.register
def _(obj: GenerationSurface) -> dict[str, Any]:
    # json safe state
    return {
        "pdgid": int(obj.pdgid.value),
        "nevents": float(obj.nevents),
        "power_law": {"cls": type(obj.power_law).__name__, "params": to_dict(obj.power_law)},
        "spatial": {"cls": type(obj.spatial).__name__, "params": to_dict(obj.spatial)},
    }


@to_dict.register
def _(obj: CompositeSurface) -> dict[str, Any]:
    # store flattened list of serialized surfaces
    # init will rebuild
    return {"components": [to_dict(s) for lst in obj.components.values() for s in lst]}


# from_dict

def _leaf_from_dict(cls: type, state: dict[str, float]) -> Any:
    """Rebuild a PowerLaw, CylinderBase or CircleInjector"""
    # ensure correct types
    for k, v in state.items():
        if isinstance(v, bool) or not isinstance(v, (float, int)):
            raise TypeError(f"{cls.__name__}.from_dict: '{k}' must be a number, got {type(v).__name__}")

    # rely on init to validate the rest
    return cls(**state)


def powerlaw_from_dict(state: dict[str, float]) -> PowerLaw:
    """Rebuild a PowerLaw from its serialized params."""
    return _leaf_from_dict(PowerLaw, state)


def spatial_from_dict(cls: type, state: dict[str, float]) -> Any:
    """Rebuild a spatial distribution of the given class from its serialized params."""
    return _leaf_from_dict(cls, state)


def surface_from_dict(state: Mapping[str, Any]) -> GenerationSurface:
    """Rebuild a GenerationSurface from its serialized state."""
    cls = GenerationSurface

    # ensure required params are included
    # need to check explicitly as we have to rebuild powerlaw and spatial objects before initializing
    required = ("power_law", "spatial", "pdgid", "nevents")
    missing = [param for param in required if param not in state]
    if missing:
        raise TypeError(f"{cls.__name__}.from_dict: missing required keys {missing}, got {sorted(state)}")

    # ensure nevents is a float or int
    nevents = state["nevents"]
    if isinstance(nevents, bool) or not isinstance(nevents, (int, float)):
        raise TypeError(f"{cls.__name__}.from_dict: 'nevents' must be a number, got {type(nevents).__name__}")

    # ensure pdgid is an int (enumification validates the int is valid later)
    pdgid = state["pdgid"]
    if isinstance(pdgid, bool) or not isinstance(pdgid, int):
        raise TypeError(f"{cls.__name__}.from_dict: 'pdgid' must be an int, got {type(pdgid).__name__}")

    # reconstruct powerlaw and spatial objects
    rebuilt_state = dict(state)
    for p, resolve in (("power_law", resolve_powerlaw), ("spatial", resolve_spatial)):
        # ensure value is a dict
        sub = state[p]
        if not isinstance(sub, dict):
            raise TypeError(f"{cls.__name__}.from_dict: '{p}' must be a dict, got {type(sub).__name__}")
        if set(sub) != {"cls", "params"}:
            raise TypeError(f"{cls.__name__}.from_dict: '{p}' must have keys 'cls' and 'params', got {sorted(sub)}")

        # make sure class name is a str
        name = sub["cls"]
        if not isinstance(name, str):
            raise TypeError(f"{cls.__name__}.from_dict: '{p}.cls' must be a str, got {type(name).__name__}")

        # make sure params is a dict
        params = sub["params"]
        if not isinstance(params, dict):
            raise TypeError(f"{cls.__name__}.from_dict: '{p}.params' must be a dict, got {type(params).__name__}")

        # resolver rejects unknown names
        # class itself validates params
        rebuilt_state[p] = _leaf_from_dict(resolve(name), params)

    # rely on init to validate rest
    return cls(**rebuilt_state)


def composite_from_dict(state: dict[str, Any]) -> CompositeSurface:
    """Rebuild a CompositeSurface from its serialized state."""
    cls = CompositeSurface

    # ensure all required keys exist
    required = ("components",)
    missing = [param for param in required if param not in state]
    if missing:
        raise TypeError(f"{cls.__name__}.from_dict: missing required keys {missing}, got {sorted(state)}")

    # ensure components is a list
    components = state["components"]
    if not isinstance(components, list):
        raise TypeError(f"{cls.__name__}.from_dict: 'components' must be a list, got {type(components).__name__}")

    # rebuild each surface
    surfaces = []
    for i, surface_dict in enumerate(state["components"]):
        # ensure surface_dict is a dict
        if not isinstance(surface_dict, dict):
            raise TypeError(
                f"{cls.__name__}.from_dict: 'components' must be a list of dicts, "
                f"got {type(surface_dict).__name__} at index {i}"
            )

        # class itself validates surface_dict
        surfaces.append(surface_from_dict(surface_dict))

    return cls(*surfaces)