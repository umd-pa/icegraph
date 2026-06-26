# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

__all__ = ["SimweighterConfig"]


class SimweighterConfig(BaseModel):
    flux:           FluxConfig
    weighter:       str
    out:            str
    weight_group:   str


class FluxConfig(BaseModel):
    name:   str
    kwargs: dict[str, Any]
