# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import TypeVar, cast, ClassVar, Any, final, Generic, TYPE_CHECKING
from abc import abstractmethod

from icegraph.types.plugins import Plugin

from .types import ServiceView, ServiceContext

__all__ = ["Service"]


V = TypeVar("V", bound=ServiceView)
C = TypeVar("C")


class Service(Plugin[C, ServiceContext], Generic[V, C]):
    # any service dependencies this service has
    deps: ClassVar[tuple[str, ...]] = tuple()

    # stable surface to expose to components and other services
    interface: ClassVar[type[ServiceView]]

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()

        if getattr(cls, "interface", None) is None:
            raise RuntimeError(f"Service {cls.__name__} must implement the class variable 'interface'")

        if not isinstance(cls.deps, tuple) or not all(isinstance(d, str) for d in cls.deps):
            raise RuntimeError(f"Dependencies for service {cls.__name__} must be a tuple of str.")

    @abstractmethod
    def load_state_dict(self, state: dict[str, Any]) -> None:
        ...

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        ...

    @final
    def view(self) -> V:
        cls = type(self)
        if not isinstance(self, cls.interface):
            raise TypeError(
                f"'{cls.__name__}' does not satisfy dependencies of surface '{cls.interface.__name__}'."
            )
        return cast(V, self)
