# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import Any

# local package
from icegraph.engine.components.factory import ComponentFactoryBase

# local subpackage
from .model import Model

# implementations
from . import variants

__all__ = ["ModelFactory"]


class ModelFactory(ComponentFactoryBase[Model[Any]]):
    pass


for name in variants.__all__:
    ModelFactory.register(getattr(variants, name))
