# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Mapping, Iterator, Any, TypeVar, overload, Literal, TYPE_CHECKING
from dataclasses import dataclass, field

from icegraph.common.engine import ComponentKind

from .component import Component
from .types import ComponentContext
from .config import ComponentConfig
from .factory import ComponentFactoryBase

# import each built in component
from .edges import EdgeBuilder, EdgeBuilderFactory
from .loss import LossFunction, LossFactory
from .model import Model, ModelFactory
from .normalizer import Normalizer, NormalizerFactory
from .optimizer import Optimizer, OptimizerFactory
from .transformer import Transformer, TransformerFactory

if TYPE_CHECKING:
    from ..services import ServiceManager

    from icegraph.engine.policy import Policy

    from ..engine import Engine, EngineConfig

import logging
logger = logging.getLogger(__name__)

__all__ = ["ComponentManager"]


_BUILD_ORDER: tuple[ComponentKind, ...] = (
    ComponentKind.MODEL,
    ComponentKind.TRANSFORMER,
    ComponentKind.NORMALIZER,
    ComponentKind.EDGES,
    ComponentKind.LOSS,
    ComponentKind.OPTIMIZER,
)


E = TypeVar("E", bound="Engine[EngineConfig]")


@dataclass
class ComponentManager(Mapping[ComponentKind, Component[Any]]):
    _components: dict[ComponentKind, Component[Any]] = field(default_factory=dict)

    def __getitem__(self, component: ComponentKind) -> Component[Any]:
        return self._components[component]

    def __iter__(self) -> Iterator[ComponentKind]:
        yield from self._components

    def __len__(self) -> int:
        return len(self._components)

    @overload
    def require(self, component: Literal[ComponentKind.EDGES], *, required_by: type[Any] | None = None) -> EdgeBuilder[Any]: ...
    @overload
    def require(self, component: Literal[ComponentKind.LOSS], *, required_by: type[Any] | None = None) -> LossFunction[Any]: ...
    @overload
    def require(self, component: Literal[ComponentKind.MODEL], *, required_by: type[Any] | None = None) -> Model[Any]: ...
    @overload
    def require(self, component: Literal[ComponentKind.NORMALIZER], *, required_by: type[Any] | None = None) -> Normalizer[Any]: ...
    @overload
    def require(self, component: Literal[ComponentKind.OPTIMIZER], *, required_by: type[Any] | None = None) -> Optimizer[Any]: ...
    @overload
    def require(self, component: Literal[ComponentKind.TRANSFORMER], *, required_by: type[Any] | None = None) -> Transformer[Any]:...
    @overload
    def require(self, component: ComponentKind, *, required_by: type[Any] | None = None) -> Component[Any]: ...

    def require(self, component: ComponentKind, *, required_by: type[Any] | None = None) -> Component[Any]:
        value = self._components.get(component)

        if value is None:
            who = required_by.__name__ if required_by is not None else "<unknown>"
            raise KeyError(
                f"The component '{component}' was requested by '{who}', but has "
                f"not been initialized or does not exist."
            )

        return value

    def register(
            self,
            name: ComponentKind,
            component: Component[Any],
            *,
            overwrite: bool = False,
    ) -> None:
        if not overwrite and name in self._components:
            raise KeyError(
                f"A component is already registered under '{name}'. Pass "
                f"overwrite=True to replace it."
            )

        self._components[name] = component

    @staticmethod
    def _get_component_factory(kind: ComponentKind) -> type[ComponentFactoryBase[Component[Any]]]:
        return {
            ComponentKind.MODEL: ModelFactory,
            ComponentKind.NORMALIZER: NormalizerFactory,
            ComponentKind.TRANSFORMER: TransformerFactory,
            ComponentKind.EDGES: EdgeBuilderFactory,
            ComponentKind.OPTIMIZER: OptimizerFactory,
            ComponentKind.LOSS: LossFactory
        }[kind]

    @classmethod
    def from_config(
            cls,
            config: dict[ComponentKind, ComponentConfig], *,
            services: ServiceManager,
            debug: bool,
            policy: Policy | None = None,
            state_dicts: dict[str, dict[str, Any]] | None = None
    ) -> ComponentManager:
        # iteratively construct manager
        components = cls()

        # ensure config provides keys present in build order
        extra = set(config.keys()) - set(_BUILD_ORDER)
        if extra:
            logger.warning(
                "%s: got extra component keys in config not present in _BUILD_ORDER: %s. These keys will be ignored.",
                cls.__name__, str(list(extra))
            )

        # build in preconfigured order
        for kind in _BUILD_ORDER:
            if kind not in config:
                continue

            c = config[kind]

            factory = cls._get_component_factory(kind)
            component = factory.create(c.name, **c.kwargs)

            # this components checkpoint, if any
            state = state_dicts.get(kind) if state_dicts is not None else None

            # preload state
            if state is not None:
                component.on_preload(dict(state))

            # run attach phase
            ctx = ComponentContext(
                services=services,
                components=components,
                contract=policy.get_contract_for(kind) if policy is not None else None,
                debug=debug
            )
            component.attach(ctx)

            # full load state
            if state is not None:
                component.load_state_dict(state)

            component.to_device()
            components.register(kind, component)

            logger.info(f"built component={kind}")

        # run binds
        components = cls._bind(components, services)

        return components

    @classmethod
    def _bind(cls, components: ComponentManager, services: ServiceManager) -> ComponentManager:
        # need to bind model to state if present
        try:
            model = components.require(ComponentKind.MODEL, required_by=cls)
        except KeyError:
            # no model in this run, so no need to bind
            return components

        state = services.require("state", required_by=cls)

        # perform bind
        model = state.bind_model(model)

        # overwrite model with bound
        # bound model is techincally not Component[Any] so this needs to be fixed
        components.register(ComponentKind.MODEL, model, overwrite=True)  # pyright: ignore[reportArgumentType]

        return components

    def close(self) -> None:
        for component in self._components.values():
            component.close()
