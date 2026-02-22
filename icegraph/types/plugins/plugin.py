# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

import inspect
from typing import Generic, TypeVar, ClassVar, Any, final
from abc import abstractmethod, ABC

from .context import PluginContext

__all__ = ["Plugin"]


C = TypeVar("C")
X = TypeVar("X", bound=PluginContext)

class Plugin(Generic[C, X], ABC):
    name: ClassVar[str]
    version: ClassVar[int]

    def __init__(self, config: C) -> None:
        super().__init__()

        # stash config
        self.config: C = config

        # cache for context
        self._ctx: X | None = None

        # run build routine
        self._built: bool = False
        self.build()
        self._built = True

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()

        if inspect.isabstract(cls):
            return

        # ensure name is defined for all subclasses
        if getattr(cls, "name", None) is None:
            raise RuntimeError(f"Plugin {cls.__name__} must implement the class variable 'name'")

        # ensure version is defined for all subclasses
        if getattr(cls, "version", None) is None:
            raise RuntimeError(f"Plugin {cls.__name__} must implement the class variable 'version'")

    @abstractmethod
    def build(self) -> None:
        ...

    @classmethod
    @abstractmethod
    def validate_config(cls, config: dict[str, Any]) -> C:
        ...

    @final
    def attach(self, ctx: X) -> None:
        """Attach this module given a context object, called by the plugin parent."""
        if self._ctx is not None:
            raise RuntimeError(f"{type(self).__name__} is already attached.")

        if not self._built:
            raise RuntimeError(f"{type(self).__name__} must be built before being attached.")

        # stash context
        self._ctx = ctx

        # call hook for downstream post-attach logic
        self.on_attach()

    def on_attach(self) -> None:
        """Called just after plugin is attached."""
        return

    def close(self) -> None:
        pass