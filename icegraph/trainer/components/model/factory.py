# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

# local package
from icegraph.types.factory import PluginFactory

# local subpackage
from .model import Model

# implementations
from .variants import GravNet

__all__ = ["ModelFactory"]


class ModelFactory(PluginFactory[Model]):
    pass


for module in [GravNet]:
    ModelFactory.register(module)
