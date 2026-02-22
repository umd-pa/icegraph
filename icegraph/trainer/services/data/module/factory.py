# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

# local package
from icegraph.types.factory import PluginFactory

# local subpackage
from .module import Module

# implementations
from .variants import GraphModule

__all__ = ["ModuleFactory"]


class ModuleFactory(PluginFactory[Module]):
    pass


# register each internal module
for module in [GraphModule]:
    ModuleFactory.register(module)
