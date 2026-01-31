# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from dataclasses import dataclass

from .spaces import TransformSpace

__all__ =["TransformSpec"]


@dataclass(frozen=True)
class TransformSpec:
    space: TransformSpace
    base: int = 10  # ignored for linear
