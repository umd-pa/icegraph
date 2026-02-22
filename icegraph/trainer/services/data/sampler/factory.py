# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

# local package
from icegraph.types.factory import PluginFactory

# local subpackage
from .sampler import Sampler

# implementations
from .variants import BlockwiseSampler

__all__ = ["SamplerFactory"]


class SamplerFactory(PluginFactory[Sampler]):
    pass


# register each internal module
for module in [BlockwiseSampler]:
    SamplerFactory.register(module)
