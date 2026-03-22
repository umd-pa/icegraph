# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

# local package
from icegraph.types.factory import PluginFactory

# local subpackage
from .normalizer import Normalizer

# implementations
from .variants import MinMax, ZScore, Centering, UnitVariance

__all__ = ["NormalizerFactory"]


class NormalizerFactory(PluginFactory[Normalizer]):
    pass


# register each internal module
for module in [MinMax, ZScore, Centering, UnitVariance]:
    NormalizerFactory.register(module)
