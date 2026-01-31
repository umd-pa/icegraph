# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import Protocol, TYPE_CHECKING

from ..types import ServiceView

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["StrategyView"]


### VIEWS

class StrategyView(ServiceView, Protocol):
    mode: str
    in_channels: int
    out_channels: int

    def adapt_targets(self, targets: Tensor) -> Tensor: ...
