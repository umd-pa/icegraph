# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import TypeAlias, Literal

from enum import Enum

__all__ = ["TransformSpace", "TransformSpaceType"]


TransformSpaceType: TypeAlias = Literal["linear", "log", "asinh"]

class TransformSpace(Enum):
    LINEAR = "linear"
    LOG = "log"
    ASINH = "asinh"

    @classmethod
    def all(cls) -> tuple[TransformSpace, ...]:
        return tuple(cls)

    @classmethod
    def values(cls) -> list[str]:
        return [space.value for space in cls.all()]

    @classmethod
    def names(cls) -> list[str]:
        return [space.name for space in cls.all()]

    @classmethod
    def non_linear(cls) -> list[TransformSpace]:
        return [space for space in cls.all() if space != cls.LINEAR]

    def format_repr(self, s: str, /) -> str:
        """Convenience to wrap a value with a latex representation of the space."""
        if self == TransformSpace.LOG:
            return rf"\log({s})"
        elif self == TransformSpace.ASINH:
            return rf"\mathrm{{asinh}}({s})"

        # if linear, just return str
        return s
