# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Any, Callable
from dataclasses import dataclass

from torch import Tensor

from icegraph.common.plugins import PluginContext

from ..services import ServiceManager

__all__ = ["ComponentContext", "ContractComponentContext", "ComponentContract"]


@dataclass(frozen=True)
class ComponentContext(PluginContext):
    services: ServiceManager
    debug: bool


@dataclass(frozen=True)
class ContractComponentContext(ComponentContext):
    contract: ComponentContract


@dataclass(frozen=True)
class ComponentContract:
    kwargs: dict[str, Any]
    forward_validator: Callable[[Tensor, bool], None]
