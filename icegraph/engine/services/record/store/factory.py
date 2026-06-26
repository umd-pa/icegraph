# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

# local package
from icegraph.common.factory import PluginFactory

# local subpackage
from .store import Store

# implementations
from . import variants

__all__ = ["StoreFactory"]


class StoreFactory(PluginFactory[Store]):
    pass


# register each internal module
for name in variants.__all__:
    StoreFactory.register(getattr(variants, name))
