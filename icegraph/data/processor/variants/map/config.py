# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

__all__ = ["MapConfig"]


class MapConfig(BaseModel):
    col:    str | int
    map_:   dict[Any, Any]      = Field(alias="map")
    strict: bool                = True
    out:    str | int | None    = None

