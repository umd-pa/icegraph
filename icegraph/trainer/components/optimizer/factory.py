# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

# local package
from icegraph.types.factory import PluginFactory

# local subpackage
from .optimizer import Optimizer

# implementations
from .standard import AdamW, SGD

__all__ = ["OptimizerFactory"]


class OptimizerFactory(PluginFactory[Optimizer]):
    pass


# register each internal module
for module in [AdamW, SGD]:
    OptimizerFactory.register(module)
