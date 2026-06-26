# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from enum import Enum

__all__ = ["AttributeDomain"]


class AttributeDomain(Enum):
    GLOBAL  = "global"
    LOCAL   = "local"

    @classmethod
    def all(cls) -> tuple[AttributeDomain, ...]:
        return tuple(cls)
