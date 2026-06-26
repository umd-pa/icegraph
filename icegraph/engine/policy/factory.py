# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import Any

# local package
from icegraph.common.factory import PluginFactory

# local subpackage
from .policy import Policy

# implementations
from . import variants

__all__ = ["PolicyFactory"]


class PolicyFactory(PluginFactory[Policy[Any]]):
    pass


# register each internal module
for name in variants.__all__:
    PolicyFactory.register(getattr(variants, name))
