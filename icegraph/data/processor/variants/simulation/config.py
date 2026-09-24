# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ...types import Columns

__all__ = ["I3SimulationConfig"]


class I3SimulationConfig(BaseModel):
    simulation:     Literal["nugen", "corsika"] = Field(alias="type")
    ids:            Columns
    to:             str                         = "generation"

    # discovered from the generation info when not given
    sim_code:       int | None                  = None

    # table keys
    weight_dict:    str                         = "I3MCWeightDict"
    corsika_info:   str                         = "I3CorsikaInfo"
    primary:        str                         = "PolyplopiaPrimary"
