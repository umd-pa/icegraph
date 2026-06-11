# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import Any

# local package
from icegraph.common.factory import PluginFactory

# local subpackage
from .normalizer import Normalizer

# implementations
from . import variants

__all__ = ["NormalizerFactory"]


class NormalizerFactory(PluginFactory[Normalizer[Any]]):
    pass


# register each internal module
for name in variants.__all__:
    NormalizerFactory.register(getattr(variants, name))
