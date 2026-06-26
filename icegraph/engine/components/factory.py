# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import Any, TypeVar

# local package
from icegraph.common.factory import PluginFactory

# local subpackage
from .component import Component

__all__ = ["ComponentFactoryBase"]


_CMPT = TypeVar("_CMPT", bound=Component[Any])


class ComponentFactoryBase(PluginFactory[_CMPT]):
    pass
