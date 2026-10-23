# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from pydantic import BaseModel

from ...types import Columns

__all__ = ["DOMConfig"]


class DOMConfig(BaseModel):
    string: str
    om:     str
    pmt:    str
    out:    str | list[str] = ["x", "y", "z"]
