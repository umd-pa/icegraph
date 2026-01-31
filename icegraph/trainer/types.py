# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Any, Self, Mapping, Iterator, TYPE_CHECKING, Protocol
from dataclasses import dataclass

if TYPE_CHECKING:
    from .trainer import Trainer
    from .services import ServiceManager

__all__ = ["AttachContext", "Params"]


### CONTEXT

@dataclass(frozen=True)
class AttachContext:
    services: ServiceManager


### PARAMS

@dataclass(frozen=True)
class Params(Mapping[str, Any]):
    _data: Mapping[str, Any]
    _consumer: str

    def __iter__(self) -> Iterator[str]:
        yield from self._data

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, item: str) -> Any:
        value = self._data[item]

        # if a dict with str key, return as new Params
        if isinstance(value, dict) and all(isinstance(k, str) for k in value.keys()):
            return Params(value, self._consumer)

        return value

    def require(self, key: str) -> Any:
        value = self._data.get(key)

        if value is None:
            raise RuntimeError(f"{self._consumer}: requires key word argument '{key}'.")

        return value

    def to_struct(self) -> dict[str, Any]:
        return {"_data": self._data, "_consumer": self._consumer}

    @classmethod
    def from_struct(cls, struct: dict[str, Any]) -> Self:
        return cls(_data=struct["_data"], _consumer=struct["_consumer"])

    @classmethod
    def empty(cls) -> Self:
        return cls({}, "EmptyParams")
