# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from enum import Enum

__all__ = ["TransformSpace"]


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
