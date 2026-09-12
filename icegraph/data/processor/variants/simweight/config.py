# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ...types import Columns

__all__ = ["SimWeightConfig"]


def _default_cols() -> list[str]:
    # the mixture independent generation quantities, everything else needed to
    # weight an event is carried by the generation surface
    return ["energy", "cos_zen", "pdgid", "event_weight"]


class SimWeightConfig(BaseModel):
    weighter:   Literal["nugen", "corsika", "genie"]
    tables:     Columns
    ids:        Columns

    # number of files the generation surface is normalized to (always 1 here), the
    # surfaces are summed over shards at load time. set to null for simulation
    # carrying S-frames where simweights derives the surface from the file itself
    nfiles:     Literal[1] | None = 1

    cols:       Columns     = Field(default_factory=_default_cols)
    to:         str         = "generation"
    attr:       str         = "surface"
