# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Mapping, Iterator, Self, Any, overload, Literal, TypedDict
from dataclasses import dataclass, field

from icegraph.common.engine import Engine

from .component import Component
from .types import ComponentContext, ContractComponentContext

# import each built in component
from .adapter import Adapter, AdapterFactory, AdapterContext
from .loss import LossFunction, LossFactory, LossContext
from .model import Model, ModelFactory, ModelContext
from .normalizer import Normalizer, NormalizerFactory, NormalizerContext
from .optimizer import Optimizer, OptimizerFactory, OptimizerContext
from .transformer import Transformer, TransformerFactory, TransformerContext

__all__ = ["ComponentManager"]


ComponentSpec = TypedDict("ComponentSpec", {"name": str, "kwargs": dict[str, Any]})


@dataclass
class ComponentManager(Mapping[str, Component[Any, ComponentContext]]):
    _components: dict[str, Component[Any, ComponentContext]] = field(default_factory=dict)

    def __getitem__(self, component: str) -> Component[Any, ComponentContext]:
        return self._components[component]

    def __iter__(self) -> Iterator[str]:
        yield from self._components

    def __len__(self) -> int:
        return len(self._components)

    @overload
    def require(self, service: Literal["adapter"], *, required_by: type[Any] | None = None) -> Adapter[Any]: ...
    @overload
    def require(self, service: Literal["loss"], *, required_by: type[Any] | None = None) -> LossFunction[Any]: ...
    @overload
    def require(self, service: Literal["model"], *, required_by: type[Any] | None = None) -> Model[Any]: ...
    @overload
    def require(self, service: Literal["normalizer"], *, required_by: type[Any] | None = None) -> Normalizer[Any]: ...
    @overload
    def require(self, service: Literal["optimizer"], *, required_by: type[Any] | None = None) -> Optimizer[Any]: ...
    @overload
    def require(self, service: Literal["transformer"], *, required_by: type[Any] | None = None) -> Transformer[Any]:...
    @overload
    def require(self, service: str, *, required_by: type[Any] | None = None) -> Component[Any, ComponentContext]: ...

    def require(self, service: str, *, required_by: type[Any] | None = None) -> Component[Any, ComponentContext]:
        value = self._services.get(service)

        if value is None:
            who = required_by.__name__ if required_by is not None else "<unknown>"
            raise KeyError(
                f"The component '{service}' was requested by '{who}', but has "
                f"not been initialized or does not exist."
            )

        return value

    @classmethod
    def from_config(
            cls,
            engine: Engine,
            config: dict[str, ComponentSpec]
    ) -> Self:
        if "adapter" not in config:
            raise RuntimeError(f"Adapter configurations are required regardless of Engine.")

        # iteratively construct manager
        instance = cls()

        # start with the adapter
        adapter = cls._adapter_constructor(
            config["adapter"]["name"], config["adapter"]["kwargs"]
        )
        instance._components["adapter"] = adapter

        # build the rest if required
        if "transformer" in config:
            instance._components["transformer"] = cls._transformer_constructor(
                config["transformer"]["name"], config["transformer"]["kwargs"], instance._components["adapter"]
            )

        return instance

    @staticmethod
    def _adapter_constructor(name: str, kwargs: dict[str, Any]) -> Adapter[Any]:

    @staticmethod
    def _optimizer_constructor(name: str, kwargs: dict[str, Any], adapter: Adapter[Any]) -> Optimizer[Any]:

    @staticmethod
    def _normalizer_constructor(name: str, kwargs: dict[str, Any], adapter: Adapter[Any]) -> Normalizer[Any]:

    @staticmethod
    def _transformer_constructor(name: str, kwargs: dict[str, Any], adapter: Adapter[Any]) -> Transformer[Any]:

    @staticmethod
    def _model_constructor(name: str, kwargs: dict[str, Any], adapter: Adapter[Any]) -> Model[Any]:

    @staticmethod
    def _loss_constructor(name: str, kwargs: dict[str, Any], adapter: Adapter[Any]) -> LossFunction[Any]:

    def close(self) -> None:
        for component in self._components.values():
            component.close()
