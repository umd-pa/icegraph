# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from ..types import ServiceView

from .types import ComputedMetric

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["MetricView"]


### VIEW

class MetricView(ServiceView, Protocol):
    def compute(self) -> list[ComputedMetric]: ...
    def update(self, out: Tensor, target: Tensor) -> None: ...
    def update_summaries(self) -> None: ...
    def reset(self) -> None: ...
