# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Any

# local package
from icegraph.common.factory import PluginFactory

# local subpackage
from .processor import Processor

# implementations
from . import variants

__all__ = ["ProcessorFactory"]


class ProcessorFactory(PluginFactory[Processor[Any]]):
    pass


for name in variants.__all__:
    ProcessorFactory.register(getattr(variants, name))
