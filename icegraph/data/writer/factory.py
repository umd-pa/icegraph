# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Any

# local package
from icegraph.types.factory import PluginFactory

# local subpackage
from .writer import Writer

# implementations
from . import variants

__all__ = ["WriterFactory"]


class WriterFactory(PluginFactory[Writer[Any]]):
    pass


for name in variants.__all__:
    WriterFactory.register(getattr(variants, name))
