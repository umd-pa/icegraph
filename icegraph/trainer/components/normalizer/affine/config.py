# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from pydantic import BaseModel

from icegraph.types.transforms import TransformSpaceType

__all__ = ["AffineNormalizerConfig"]


class AffineNormalizerConfig(BaseModel):
    transforms: dict[str, TransformSelection]


class TransformSelection(BaseModel):
    space: TransformSpaceType
    base: int = 10
