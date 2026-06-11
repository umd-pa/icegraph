# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import Any

# local package
from icegraph.common.factory import PluginFactory

# local subpackage
from .loss import LossFunction

# implementations
from . import variants

__all__ = ["LossFactory"]


class LossFactory(PluginFactory[LossFunction[Any]]):
    pass


# register each internal module
for name in variants.__all__:
    LossFactory.register(getattr(variants, name))
