# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from abc import abstractmethod
from typing import TypeVar

from torch import Tensor

from ..component import Component

from .types import ModelContext

__all__ = ["Model"]


C = TypeVar("C")


class Model(Component[C, ModelContext]):

    @abstractmethod
    def forward(self, t: Tensor, /, batch: Tensor | None = None) -> Tensor:
        ...
