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

    # compatibility, empty indicates full compatibility
    compatible: ClassVar[tuple[str, ...]] = tuple()

    _ctx: X

    def __init__(self, config: C) -> None:
        super().__init__()

        # stash config
        self.config: C = config

        # run build routine
        self._built: bool = False
        self._internal_build_before_user()
        self.build()
        self._internal_build_after_user()
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
        """
        Public plugin build hook.

        Plugin authors should put user-defined initialization logic here.
        """
        ...

    def _internal_build_before_user(self) -> None:
        """
        Internal build hook.

        This is called before `build()` and is not part of the public plugin
        API. Subclasses should not override this unless they are part of
        the core framework implementation.
        """
        return

    def _internal_build_after_user(self) -> None:
        """
        Internal build hook.

        This is called after `build()` and is not part of the public plugin
        API. Subclasses should not override this unless they are part of
        the core framework implementation.
        """
        return

    @classmethod
    @abstractmethod
    def validate_config(cls, config: dict[str, Any]) -> C:
        ...

    @property
    def is_attached(self) -> bool:
        return getattr(self, "_ctx", None) is not None

    @final
    def attach(self, ctx: X) -> None:
        """Attach this module given a context object, called by the plugin parent."""
        if getattr(self, "_ctx", None) is not None:
            raise RuntimeError(f"{type(self).__name__} is already attached.")

        if not self._built:
            raise RuntimeError(f"{type(self).__name__} must be built before being attached.")

        # stash context
        self._ctx = ctx

        # call hook for downstream post-attach logic
        self.on_attach()
        self._internal_on_attach_after_user()

    def on_attach(self) -> None:
        """Called just after plugin is attached."""
        return

    def _internal_on_attach_after_user(self) -> None:
        return

    def close(self) -> None:
        pass