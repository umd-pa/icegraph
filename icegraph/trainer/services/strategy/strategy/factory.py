# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

# local package
from icegraph.types.factory import PluginFactory

# local subpackage
from .strategy import Strategy

# implementations
from .variants import Regression, Multiclass

__all__ = ["StrategyFactory"]


class StrategyFactory(PluginFactory[Strategy]):
    pass


# register each internal module
for module in [Regression, Multiclass]:
    StrategyFactory.register(module)
