# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from pydantic import Field

from ...config import Config

__all__ = ["SumInQuadratureConfig"]


class SumInQuadratureConfig(Config):
    # logical feature column groups summed in quadrature to form the edge weight
    weight_cols: list[str] = Field(min_length=1)
