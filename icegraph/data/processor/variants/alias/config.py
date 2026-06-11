# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from pydantic import BaseModel, Field

__all__ = ["Config"]


class Config(BaseModel):
    map_: dict[str | int, list[str] | list[int]] = Field(alias="map")
