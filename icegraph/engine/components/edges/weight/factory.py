# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import Any

from icegraph.common.factory import PluginFactory

from .weight import EdgeWeight

# implementations
from . import variants

__all__ = ["EdgeWeightFactory"]


class EdgeWeightFactory(PluginFactory[EdgeWeight[Any]]):
    pass


# register each internal module
for name in variants.__all__:
    EdgeWeightFactory.register(getattr(variants, name))
