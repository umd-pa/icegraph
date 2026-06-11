# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Any, TypeVar

from icegraph.common.plugins import Plugin

from .exceptions import UnknownModuleError
from .factory import Factory

__all__ = ["PluginFactory"]


P = TypeVar("P", bound=Plugin[Any, Any])

class PluginFactory(Factory[P]):

    @classmethod
    def create(cls, name: str, **config: Any) -> P:
        """
        Instantiate a registered module. Raises UnknownModuleError if the name is unknown.
        """
        # normalize to lower case
        # this is so config via factories is not case-sensitive
        name = name.lower()

        try:
            spec = cls._typed_registry()[name]
        except KeyError:
            # raise if not registered
            raise UnknownModuleError(name, list(cls._typed_registry()))

        return spec(spec.validate_config(config))
