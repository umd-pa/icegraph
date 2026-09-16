# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from ...types import Columns

__all__ = ["SimWeightConfig"]


class SimWeightConfig(BaseModel):
    simulation: Literal["nugen", "corsika", "genie"]
    tables:     Columns
    ids:        Columns
    to:         str         = "generation"
    attr:       str         = "surface"
