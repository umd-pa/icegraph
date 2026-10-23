# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from pydantic import BaseModel, Field

__all__ = ["GaussianConfig"]


class GaussianConfig(BaseModel):
    # logical feature column groups the endpoint distance is taken over
    cols: list[str] = Field(min_length=1)

    # width of the kernel, in the units of cols
    sigma: float = Field(gt=0)
