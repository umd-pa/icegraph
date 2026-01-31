# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import Protocol, runtime_checkable, TYPE_CHECKING, Any
from dataclasses import dataclass

from icegraph.trainer.types import AttachContext

if TYPE_CHECKING:
    from ..trainer import Trainer

__all__ = ["ServiceView", "ServiceContext"]


@runtime_checkable
class ServiceView(Protocol):
    """Marker base protocol for dependency surfaces."""
    def close(self) -> None: ...
    def load_state_dict(self, state: dict[str, Any]) -> None: ...
    def state_dict(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class ServiceContext(AttachContext):
    trainer: Trainer
