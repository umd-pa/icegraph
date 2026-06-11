# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

# local package
from icegraph.common.factory import PluginFactory

# local subpackage
from .decoder import AttributeDecoder

# implementations
from . import variants

__all__ = ["AttributeDecoderFactory"]


class AttributeDecoderFactory(PluginFactory[AttributeDecoder]):
    pass


# register each internal module
for name in variants.__all__:
    AttributeDecoderFactory.register(getattr(variants, name))
