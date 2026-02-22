# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Mapping, Iterator, Self, Any, overload, Literal, TYPE_CHECKING
from dataclasses import dataclass, field
from graphlib import TopologicalSorter, CycleError

from .factory import ServiceFactory
from .service import Service

# import each built in service view for overloads
from .state import StateView
from .strategy import StrategyView
from .metrics import MetricView
from .data import DataView

if TYPE_CHECKING:
    from ..trainer import Trainer

    from .types import ServiceView, ServiceContext

__all__ = ["ServiceManager"]


@dataclass
class ServiceManager(Mapping[str, Service]):
    _services: dict[str, Service] = field(default_factory=dict)

    def __getitem__(self, service: str) -> Service:
        return self._services[service]

    def __iter__(self) -> Iterator[str]:
        yield from self._services

    def __len__(self) -> int:
        return len(self._services)

    @overload
    def require(self, service: Literal["state"], *, required_by: type[Any] | None = None) -> StateView:
        ...

    @overload
    def require(self, service: Literal["data"], *, required_by: type[Any] | None = None) -> DataView:
        ...

    @overload
    def require(self, service: Literal["strategy"], *, required_by: type[Any] | None = None) -> StrategyView:
        ...

    @overload
    def require(self, service: Literal["metrics"], *, required_by: type[Any] | None = None) -> MetricView:
        ...

    @overload
    def require(self, service: str, *, required_by: type[Any] | None = None) -> ServiceView:
        ...

    def require(self, service: str, *, required_by: type[Any] | None = None) -> ServiceView:
        value = self._services.get(service)

        if value is None:
            who = required_by.__name__ if required_by is not None else "<unknown>"
            raise RuntimeError(
                f"The service '{service}' was requested by '{who}', but has "
                f"not been initialized or does not exist."
            )

        return value.view()

    @classmethod
    def from_config(cls, trainer: Trainer, config: dict[str, dict[str, Any]]) -> Self:
        # iteratively construct the service manager
        instance = cls()

        # construct all services specified in config
        for name, kwargs in config.items():
            # create and register the service
            instance._services[name] = ServiceFactory.create(name, config=kwargs)

        # validate deps
        for n, s in instance._services.items():
            for d in s.deps:
                if d not in instance._services:
                    raise ValueError(f"Service '{n}' depends on missing service '{d}'")

        # topological sort
        graph = {n: set(s.deps) for n, s in instance._services.items()}
        try:
            order = TopologicalSorter(graph).static_order()
        except CycleError as e:
            raise ValueError(f"Service dependency cycle detected: {e}") from None

        # build service context (includes refs to all services)
        context = ServiceContext(instance, trainer)
        for name in order:
            instance._services[name].attach(context)

        return instance

    def close(self) -> None:
        for service in self._services.values():
            service.close()
