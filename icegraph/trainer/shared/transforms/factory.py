# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from icegraph.types.factory import Factory

from .transform import Transform

from . import standard

__all__ = ["TransformFactory"]


class TransformFactory(Factory[Transform]):
    pass


for name in standard.__all__:
    TransformFactory.register(getattr(standard, name))
