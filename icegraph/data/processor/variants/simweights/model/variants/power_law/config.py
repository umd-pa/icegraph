# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from pydantic import BaseModel

__all__ = ["PowerLawConfig"]


class PowerLawConfig(BaseModel):
    g:      int | float
    phi0:   int | float
    e0:     int | float
