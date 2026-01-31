# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import Mapping, Iterator, Self, Any
from dataclasses import dataclass, field

from ..trainer import Trainer
from ..types import Params

from .service import Service
from .factory import ServiceFactory

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

    def require(self, service: str, *, required_by: type = type(None)) -> ViewSurface:
        value = self._services.get(service)

        if value is None:
            raise RuntimeError(
                f"The service '{service}' was requested by '{required_by.__name__}', but has not yet been initialized."
            )

        return value

    @classmethod
    def from_config(cls, trainer: Trainer, config: dict[str, Any]) -> Self:
        # iteratively construct the service manager
        instance = cls.__new__(cls)

        for name, p in config.items():
            params = Params(p, name)


