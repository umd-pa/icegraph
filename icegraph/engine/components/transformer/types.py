# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from dataclasses import dataclass

from icegraph.common.transforms import TransformSpace

__all__ = ["TransformerSpec"]


@dataclass(frozen=True)
class TransformerSpec:
    space:  TransformSpace
    base:   int
