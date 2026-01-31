# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

# local package
from icegraph.types.factory import ModuleFactory

# local subpackage
from .normalizer import Normalizer
from .standard import MinMax, ZScore, Centering, UnitVariance

__all__ = ["NormalizerFactory"]


class NormalizerFactory(ModuleFactory[str, Normalizer]):
    pass


# register each internal module
for module in [MinMax, ZScore, Centering, UnitVariance]:
    NormalizerFactory.register(module)
