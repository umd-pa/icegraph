# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from pydantic import BaseModel

from ...types import Columns

__all__ = ["KNNConfig"]


class KNNConfig(BaseModel):
    by:     Columns
    col:    str | int
    out:    Columns
    k:      int
