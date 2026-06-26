# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import Any

# local package
from icegraph.engine.components.factory import ComponentFactoryBase

# local subpackage
from .optimizer import Optimizer

# implementations
from .variants import AdamW, SGD

__all__ = ["OptimizerFactory"]


class OptimizerFactory(ComponentFactoryBase[Optimizer[Any]]):
    pass


# register each internal module
for module in [AdamW, SGD]:
    OptimizerFactory.register(module)
