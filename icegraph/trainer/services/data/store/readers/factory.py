# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

# local package
from icegraph.types.factory import PluginFactory

# local subpackage
from .reader import Reader

# implementations
from .variants import LMDB

__all__ = ["ReaderFactory"]


class ReaderFactory(PluginFactory[Reader]):
    pass


# register each internal module
for module in [LMDB]:
    ReaderFactory.register(module)
