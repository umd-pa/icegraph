# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import ClassVar, TypeVar, Generic, Any

__all__ = ["Store"]


T = TypeVar("T")

class Store(Generic[T]):
    _store: ClassVar[Any] = None

    @classmethod
    def _typed_store(cls) -> T:
        return cls._store  # type: ignore[return-value]

    @classmethod
    def register(cls, instance: T) -> None:
        """Overwrites any existing value in the store."""
        cls._store = instance

    @classmethod
    def get(cls) -> T:
        if cls._store is None:
            raise RuntimeError("No instance has been registered to the store.")
        return cls._store
