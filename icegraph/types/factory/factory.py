# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Type, Any, Protocol, ClassVar, TypeVar, Generic, Hashable

from .exceptions import UnknownModuleError

__all__ = ["ModuleFactory"]


K = TypeVar("K", bound=Hashable)

class NamedModule(Protocol[K]):
    name: K

# bound generic type
T = TypeVar("T", bound=NamedModule)


class ModuleFactory(Generic[K, T]):
    _registry: ClassVar[dict[Hashable, type[Any]]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls._registry = {}  # new registry per subclass

    @classmethod
    def _typed_registry(cls) -> dict[K, type[T]]:
        return cls._registry  # type: ignore[return-value]

    @classmethod
    def register(cls, module: Type[T]) -> None:
        """Register a module under its name."""
        if (name := getattr(module, "name")) is None:
            raise ValueError("All modules must implement the class attribute 'name'.")

        # normalize to lower case if str key
        # this is so config via factories is not case-sensitive
        name = name.lower() if isinstance(name, str) else name

        cls._typed_registry()[name] = module

    @classmethod
    def create(cls, name: K, **kwargs: Any) -> T:
        """
        Instantiate a registered module. Raises UnknownModuleError if the name is unknown.
        """
        # normalize to lower case if str key
        # this is so config via factories is not case-sensitive
        name = name.lower() if isinstance(name, str) else name

        try:
            spec = cls._typed_registry()[name]
        except KeyError:
            # raise if not registered
            raise UnknownModuleError(name, list(cls._typed_registry()))

        return spec(**kwargs)  # type: ignore[misc]
