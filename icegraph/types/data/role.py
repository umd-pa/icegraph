# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from enum import Enum

__all__ = ["ModelInputRole"]


class ModelInputRole(Enum):
    FEATURES = "features"
    LABELS = "labels"

    @classmethod
    def all(cls) -> tuple[ModelInputRole, ...]:
        return tuple(cls)

    @classmethod
    def values(cls) -> list[str]:
        return [space.value for space in cls.all()]

    @classmethod
    def names(cls) -> list[str]:
        return [space.name for space in cls.all()]
