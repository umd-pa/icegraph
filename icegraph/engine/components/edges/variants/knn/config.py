# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from pydantic import BaseModel, Field

__all__ = ["Config"]


class Config(BaseModel):
    # number of nearest neighbours to connect each node to
    k: int = Field(ge=1)

    # logical feature column groups spanning the space the neighbour search runs in
    neighbor_cols: list[str] = Field(min_length=1)
