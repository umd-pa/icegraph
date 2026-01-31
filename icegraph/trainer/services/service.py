# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import TypeVar, cast, ClassVar, Any

from ..module import TrainerModule

from .types import ServiceView, ServiceContext

__all__ = ["Service"]


V = TypeVar("V", bound=ServiceView)

class Service(TrainerModule[ServiceContext]):
    name: str

    # stable surface to expose to components and other services
    view: ClassVar[type[ServiceView]]

    # any service dependencies this service has
    deps: ClassVar[list[str]] = []

    def __init_subclass__(cls) -> None:
        for var in ["name", "surface"]:
            if getattr(cls, var, None) is None:
                raise RuntimeError(f"Service {cls.__name__} must implement the class variable '{var}'")

        if not isinstance(cls.deps, list) or not all(isinstance(i, str) for i in cls.deps):
            raise RuntimeError(f"Dependencies for service {cls.__name__} must be a list of str.")

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)

    def state_dict(self) -> dict[str, Any]:
        return {}

    def view(self) -> type[ServiceView]:
        cls = type(self)
        if not isinstance(self, cls.view):
            raise TypeError(
                f"'{cls.__name__}' does not satisfy dependencies of surface '{cls.view.__name__}'."
            )
        return cast(cls.view, self)

    def close(self) -> None:
        pass

