# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import TYPE_CHECKING, Protocol

from ..types import ServiceView

if TYPE_CHECKING:
    import torch

__all__ = ["StateView"]


class StateView(ServiceView, Protocol):
    rank:       int
    world:      int
    local_rank: int
    device:     torch.device

    def is_main_process(self) -> bool: ...
    def barrier(self) -> None: ...
    def is_ddp(self) -> bool: ...
