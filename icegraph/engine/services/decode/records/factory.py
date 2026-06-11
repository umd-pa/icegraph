# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

# local package
from icegraph.common.factory import PluginFactory

# local subpackage
from .decoder import RecordDecoder

# implementations
from . import variants

__all__ = ["RecordDecoderFactory"]


class RecordDecoderFactory(PluginFactory[RecordDecoder]):
    pass


# register each internal module
for name in variants.__all__:
    RecordDecoderFactory.register(getattr(variants, name))
