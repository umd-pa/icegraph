# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from pydantic import BaseModel

from ...types import Columns

__all__ = ["PivotConfig"]


class PivotConfig(BaseModel):
    index:  Columns
    col:    str
    values: str
