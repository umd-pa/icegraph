# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import Any

# local package
from icegraph.engine.components.factory import ComponentFactoryBase

# local subpackage
from .transformer import Transformer

# implementations
from . import variants

__all__ = ["TransformerFactory"]


class TransformerFactory(ComponentFactoryBase[Transformer[Any]]):
    pass


# register each internal module
for name in variants.__all__:
    TransformerFactory.register(getattr(variants, name))
