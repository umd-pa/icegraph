# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import TypeVar
from abc import abstractmethod

from torch import Tensor

from ..component import Component

from .types import LossContext

__all__ = ["LossFunction"]


C = TypeVar("C")


class LossFunction(Component[C, LossContext]):
    @abstractmethod
    def forward(self, out: Tensor, target: Tensor, /) -> Tensor:
        """Forward pass through the loss function."""
        ...
