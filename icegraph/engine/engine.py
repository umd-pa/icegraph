# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from abc import abstractmethod, ABC
from pathlib import Path
from typing import Generic, TypeVar, Any, Self
import json
import tomllib

import yaml

from .services import ServiceManager
from .status import Status, StatusEvent
from .components import ComponentManager
from .policy import Policy, PolicyFactory, PolicyContext
from .config import EngineConfig
from .callbacks import CallbackManager, StatusContext

__all__ = ["Engine"]


C = TypeVar("C", bound="EngineConfig")


class Engine(ABC, Generic[C]):

    # built by setup()
    services:   ServiceManager
    policy:     Policy | None
    components: ComponentManager

    def __init__(self, config: C) -> None:
        self.config = config
        self._state_dicts: dict[str, dict[str, Any]] | None = None
        self._is_setup: bool = False

        self.callbacks: CallbackManager[Self] = CallbackManager(self)

        # always present and built first, so everything after can report through it
        self.status: Status = Status(self.config.status)
        self.status.subscribe(self._forward_status)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False

    @classmethod
    @abstractmethod
    def from_config(cls, config: dict[str, Any]) -> Engine[C]:
        ...

    # allow all file loaders to derive return type from from_config

    @classmethod
    def from_yaml(cls, path: str | Path):
        with Path(path).open("r") as f:
            return cls.from_config(yaml.safe_load(f))

    @classmethod
    def from_json(cls, path: str | Path):
        with Path(path).open("r", encoding="utf-8") as f:
            return cls.from_config(json.load(f))

    @classmethod
    def from_toml(cls, path: str | Path):
        with Path(path).open("rb") as f:
            return cls.from_config(tomllib.load(f))

    def _forward_status(self, event: StatusEvent) -> None:
        self.callbacks.fire("on_status", StatusContext(engine=self, event=event))

    def _load_state_dicts(self, state_dicts: dict[str, dict[str, Any]]) -> None:
        self._state_dicts = state_dicts

    def setup(self) -> None:
        """
        Build everything the engine needs, reporting each stage to the status.

        Engines call this at the start of ``execute``. Calling it again does nothing.
        """
        if self._is_setup:
            return

        with self.status.task("Starting up"):
            self._setup()

        self._is_setup = True

    def _setup(self) -> None:
        """Build the engine's parts in dependency order. Engines override this to add their own."""
        self._setup_services()
        self._setup_components()

    def _setup_services(self) -> None:
        self.services = ServiceManager.from_config(
            self.config.services.as_mapping(),
            debug=self.config.debug,
            status=self.status
        )

        # run expensive setup
        self.services.setup()

    def _setup_components(self) -> None:
        """Build the policy, then the components it issues contracts to."""
        self.policy = None
        if self.config.policy is not None:
            self.policy = PolicyFactory.create(self.config.policy.name, **self.config.policy.kwargs)

            # attach the adapter
            self.policy.attach(PolicyContext(services=self.services, status=self.status))

        self.components = ComponentManager.from_config(
            self.config.components.as_mapping(),
            services=self.services,
            status=self.status,
            debug=self.config.debug,
            policy=self.policy,
            state_dicts=self._state_dicts
        )

    @abstractmethod
    def execute(self) -> None:
        ...

    def close(self) -> None:
        if "components" in vars(self):
            self.components.close()
        if "services" in vars(self):
            self.services.close()

        self.status.close()
