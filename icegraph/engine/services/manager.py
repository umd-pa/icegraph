# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Self, Any, overload, Literal, TYPE_CHECKING
from typing_extensions import override
from dataclasses import dataclass, field
from collections.abc import Mapping, Iterator
from graphlib import TopologicalSorter, CycleError

from .factory import ServiceFactory
from .service import Service

# import each built in service for overloads
from .state import StateService
from .metrics import MetricService
from .data import DataService
from .record import RecordService
from .decode import DecodeService
from .types import ServiceContext

if TYPE_CHECKING:
    from ..status import Status

import logging
logger = logging.getLogger(__name__)

__all__ = ["ServiceManager"]


@dataclass
class ServiceManager(Mapping[str, Service[Any]]):
    _services: dict[str, Service[Any]] = field(default_factory=dict)

    # dependency order
    _order: list[str] = field(default_factory=list)

    @override
    def __getitem__(self, service: str) -> Service[Any]:
        return self._services[service]

    @override
    def __iter__(self) -> Iterator[str]:
        yield from self._services

    @override
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
    def require(self, service: str, *, required_by: type[Any] | None = None) -> Service[Any]: ...

    def require(self, service: str, *, required_by: type[Any] | None = None) -> Service[Any]:
        value = self._services.get(service)

        if value is None:
            who = required_by.__name__ if required_by is not None else "<unknown>"
            raise KeyError(
                f"The service '{service}' was requested by '{who}', but has "
                + f"not been initialized or does not exist."
            )

        return value

    @classmethod
    def from_config(
            cls,
            config: dict[str, dict[str, Any]], *,
            debug: bool,
            status: Status
    ) -> Self:
        """Build and attach every configured service in dependency order."""
        # iteratively construct the service manager
        instance = cls()

        # build service context (includes refs to services as they are built)
        context = ServiceContext(services=instance, status=status, debug=debug)

        with status.task("Building services", total=len(config)) as task:
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
                order = list(TopologicalSorter(graph).static_order())
            except CycleError as e:
                raise ValueError(f"Service dependency cycle detected: {e}") from None

            for name in order:
                task.update(detail=name)
                instance._services[name].attach(context)
                task.advance()

        instance._order = order
        return instance

    def setup(self) -> None:
        """Run each service's up front build in dependency order."""
        for name in self._order:
            self._services[name].setup()

    def close(self) -> None:
        for service in self._services.values():
            service.close()
