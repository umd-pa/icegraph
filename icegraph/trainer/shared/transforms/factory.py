# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from icegraph.types.factory import Factory

from .variants import Asinh, Log
from .transform import Transform

__all__ = ["TransformFactory"]


class TransformFactory(Factory[Transform]):
    pass

TransformFactory.register(Asinh)
TransformFactory.register(Log)
