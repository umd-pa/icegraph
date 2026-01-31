# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import TYPE_CHECKING, Protocol

from ..types import ServiceView

if TYPE_CHECKING:
    from torch.distributed import Work

__all__ = ["StateView"]


class StateView(ServiceView, Protocol):
    rank:       int | None
    world:      int | None
    local_rank: int | None

    def is_main_process(self) -> bool: ...
    def barrier(self) -> Work | None: ...
    def is_ddp(self) -> bool: ...
