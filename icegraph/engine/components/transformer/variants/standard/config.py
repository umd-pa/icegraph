# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from pydantic import BaseModel, field_validator

from icegraph.common.transforms import TransformSpaceType

__all__ = ["TransformerConfig", "SpaceSelection"]


class TransformerConfig(BaseModel):
    transforms: dict[str, SpaceSelection]


class SpaceSelection(BaseModel):
    space:  TransformSpaceType
    base:   int = 10

    @field_validator("base")
    @classmethod
    def validate_base(cls, base: int) -> int:
        if base <= 0 or base == 1:
            raise ValueError("Log base must be positive and not equal to 1.")
        return base
