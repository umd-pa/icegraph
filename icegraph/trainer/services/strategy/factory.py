# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

# local package
from icegraph.types.factory import Factory

# local subpackage
from .strategy import Strategy
from .standard import Regression, Multiclass

__all__ = ["StrategyFactory"]


class StrategyFactory(Factory[Strategy]):
    pass


# register each internal module
for module in [Regression, Multiclass]:
    StrategyFactory.register(module)
