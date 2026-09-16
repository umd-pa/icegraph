# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, Literal, TypeAlias, TypeVar, Union

import numpy as np
from pydantic import BaseModel, Discriminator, Field, Tag

from icegraph.common.record import Scalar

__all__ = ["Dist", "SurfaceComponent", "SurfaceAttrs"]


T = TypeVar("T")


# parameters of each distribution
# declaration order here is part of the contract

class ColumnParams(BaseModel):
    colname:        Scalar[str]


class PowerLawParams(BaseModel):
    g:              Scalar[float]
    a:              Scalar[float]
    b:              Scalar[float]
    colname:        Scalar[str]


class CircleInjectorParams(BaseModel):
    radius:         Scalar[float]
    cos_zen_min:    Scalar[float]
    cos_zen_max:    Scalar[float]
    colname:        Scalar[str]


class CylinderParams(BaseModel):
    length:         Scalar[float]
    radius:         Scalar[float]
    cos_zen_min:    Scalar[float]
    cos_zen_max:    Scalar[float]
    colname:        Scalar[str]


class ColumnDist(BaseModel):
    kind:   Scalar[Literal["Column"]]
    params: ColumnParams


class PowerLawDist(BaseModel):
    kind:   Scalar[Literal["PowerLaw"]]
    params: PowerLawParams


class CircleInjectorDist(BaseModel):
    kind:   Scalar[Literal["CircleInjector"]]
    params: CircleInjectorParams


class NaturalRateCylinderDist(BaseModel):
    kind:   Scalar[Literal["NaturalRateCylinder"]]
    params: CylinderParams


class UniformSolidAngleCylinderDist(BaseModel):
    kind:   Scalar[Literal["UniformSolidAngleCylinder"]]
    params: CylinderParams


def _dist_kind(value: Any) -> str | None:
    """Which distribution a serialized one is. Read before the value is unwrapped."""
    kind = value.get("kind") if isinstance(value, Mapping) else getattr(value, "kind", None)

    return str(kind.item()) if isinstance(kind, np.ndarray) else kind


# a distribution the kind picks out, so an unknown one is rejected by name and each
# kind is held to its own parameters
Dist: TypeAlias = Annotated[
    Union[
        Annotated[ColumnDist, Tag("Column")],
        Annotated[PowerLawDist, Tag("PowerLaw")],
        Annotated[CircleInjectorDist, Tag("CircleInjector")],
        Annotated[NaturalRateCylinderDist, Tag("NaturalRateCylinder")],
        Annotated[UniformSolidAngleCylinderDist, Tag("UniformSolidAngleCylinder")],
    ],
    Discriminator(_dist_kind),
]


class SurfaceComponent(BaseModel):
    """The surface one species was generated on."""

    pdgid:      Scalar[int]
    nevents:    Scalar[float]

    # ordered
    dists:      dict[int, Dist]


class SurfaceAttrs(BaseModel):
    """The generation surface of one file, as the i3-simweight processor writes it."""

    # simulation this file was generated with
    simulation: Scalar[str]
    code:       Scalar[int]
    column:     Scalar[str]

    version:    Scalar[str] = Field(alias="__simweights_version__")

    data:       dict[int, SurfaceComponent] = Field(min_length=1)
