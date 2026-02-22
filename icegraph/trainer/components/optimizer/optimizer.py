# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import TypeVar
from abc import abstractmethod

from ..component import Component

from .types import OptimizerContext

__all__ = ["Optimizer"]


C = TypeVar("C")


class Optimizer(Component[C, OptimizerContext]):
    @abstractmethod
    def step(self) -> None:
        ...

    @abstractmethod
    def zero_grad(self, *, set_to_none: bool = True) -> None:
        ...
