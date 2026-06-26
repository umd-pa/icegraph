# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from enum import StrEnum

__all__ = ["ComponentKind"]


class ComponentKind(StrEnum):
    NORMALIZER  = "normalizer"
    MODEL       = "model"
    TRANSFORMER = "transformer"
    OPTIMIZER   = "optimizer"
    LOSS        = "loss"

    @classmethod
    def all(cls) -> tuple[ComponentKind, ...]:
        return tuple(cls)
