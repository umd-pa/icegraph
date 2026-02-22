# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from pydantic import BaseModel

from ...types import Columns

__all__ = ["StatsConfig"]


class StatsConfig(BaseModel):
    cols:   Columns
    stats:  list[str]

