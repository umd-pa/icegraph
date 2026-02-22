# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

# local package
from icegraph.types.factory import PluginFactory

# local subpackage
from .store import Store

# implementations
from .variants import LRUShardStore

__all__ = ["StoreFactory"]


class StoreFactory(PluginFactory[Store]):
    pass


# register each internal module
for module in [LRUShardStore]:
    StoreFactory.register(module)
