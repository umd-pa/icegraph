# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from abc import abstractmethod, ABC
from pathlib import Path
from typing import Generic, TypeVar, Any, Self
from functools import cached_property
import json
import tomllib

import yaml

from .services import ServiceManager
from .components import ComponentManager
from .policy import Policy, PolicyFactory, PolicyContext
from .config import EngineConfig
from .callbacks import CallbackManager

__all__ = ["Engine"]


C = TypeVar("C", bound="EngineConfig")


class Engine(ABC, Generic[C]):

    def __init__(self, config: C) -> None:
        self.config = config
        self._state_dicts: dict[str, dict[str, Any]] | None = None

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

    @cached_property
    def callbacks(self) -> CallbackManager[Self]:
        return CallbackManager(self)

    @cached_property
    def services(self) -> ServiceManager:
        return ServiceManager.from_config(
            self.config.services.as_mapping(),
            debug=self.config.debug
        )

    @cached_property
    def policy(self) -> Policy | None:
        if self.config.policy is None:
            return None

        policy = PolicyFactory.create(self.config.policy.name, **self.config.policy.kwargs)

        # attach the adapter
        ctx = PolicyContext(services=self.services)
        policy.attach(ctx)

        return policy

    @cached_property
    def components(self) -> ComponentManager:
        return ComponentManager.from_config(
            self.config.components.as_mapping(),
            services=self.services,
            debug=self.config.debug,
            policy=self.policy,
            state_dicts=self._state_dicts
        )

    def _load_state_dicts(self, state_dicts: dict[str, dict[str, Any]]) -> None:
        self._state_dicts = state_dicts

    @abstractmethod
    def execute(self) -> None:
        ...

    def close(self) -> None:
        if "components" in self.__dict__:
            self.components.close()
        if "services" in self.__dict__:
            self.services.close()