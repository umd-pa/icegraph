# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Protocol, runtime_checkable, TYPE_CHECKING, Any
from dataclasses import dataclass

from icegraph.types.plugins import PluginContext

if TYPE_CHECKING:
    from ..trainer import Trainer

    from .manager import ServiceManager

__all__ = ["ServiceView", "ServiceContext"]


@runtime_checkable
class ServiceView(Protocol):
    """Marker base protocol for dependency surfaces."""

    def close(self) -> None: ...
    def load_state_dict(self, state: dict[str, Any]) -> None: ...
    def state_dict(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class ServiceContext(PluginContext):
    services:   ServiceManager
    trainer:    Trainer
