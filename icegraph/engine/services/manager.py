# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Mapping, Iterator, Self, Any, overload, Literal
from dataclasses import dataclass, field
from graphlib import TopologicalSorter, CycleError

from icegraph.common.engine import Engine

from .factory import ServiceFactory
from .service import Service

# import each built in service for overloads
from .state import StateService
from .metrics import MetricService
from .data import DataService
from .record import RecordService
from .decode import DecodeService

from .types import ServiceContext

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
    def require(self, service: Literal["state"], *, required_by: type[Any] | None = None) -> StateService: ...
    @overload
    def require(self, service: Literal["data"], *, required_by: type[Any] | None = None) -> DataService: ...
    @overload
    def require(self, service: Literal["metrics"], *, required_by: type[Any] | None = None) -> MetricService: ...
    @overload
    def require(self, service: Literal["record"], *, required_by: type[Any] | None = None) -> RecordService: ...
    @overload
    def require(self, service: Literal["decode"], *, required_by: type[Any] | None = None) -> DecodeService: ...
    @overload
    def require(self, service: str, *, required_by: type[Any] | None = None) -> Service: ...

    def require(self, service: str, *, required_by: type[Any] | None = None) -> Service:
        value = self._services.get(service)

        if value is None:
            who = required_by.__name__ if required_by is not None else "<unknown>"
            raise KeyError(
                f"The service '{service}' was requested by '{who}', but has "
                f"not been initialized or does not exist."
            )

        return value

    @classmethod
    def from_config(cls, engine: Engine, config: dict[str, dict[str, Any]]) -> Self:
        # iteratively construct the service manager
        instance = cls()

        # construct all services specified in config
        for name, kwargs in config.items():
            # create and register the service
            instance._services[name] = ServiceFactory.create(name, **kwargs)

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
        context = ServiceContext(instance, engine.ensure_compatible)
        for name in order:
            instance._services[name].attach(context)

        return instance

    def close(self) -> None:
        for service in self._services.values():
            service.close()
