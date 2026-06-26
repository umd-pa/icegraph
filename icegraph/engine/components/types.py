# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Any, Callable, TYPE_CHECKING, TypeVar, Generic
from dataclasses import dataclass

from torch import Tensor

from icegraph.common.plugins import PluginContext

if TYPE_CHECKING:
    from ..services import ServiceManager

    from .component import Component
    from .manager import ComponentManager

__all__ = ["ComponentContext", "ComponentContract"]


_CMPT = TypeVar("_CMPT", bound="Component[Any]")


@dataclass(frozen=True)
class ComponentContract(Generic[_CMPT]):
    kwargs: dict[str, Any]
    validator: Callable[[_CMPT], None]
    forward_validator: Callable[[Tensor, bool], None] | None  # some components have no forward, like optimizer


@dataclass(frozen=True)
class ComponentContext(PluginContext):
    services: ServiceManager
    components: ComponentManager
    contract: ComponentContract | None
    debug: bool
