# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import ClassVar
from abc import ABC, abstractmethod

from torch import Tensor

__all__ = ["TransformerModule"]


class TransformerModule(ABC):
    name: ClassVar[str]

    def __init_subclass__(cls) -> None:
        if getattr(cls, "name", None) is None:
            raise ValueError(f"All subclasses of {cls.__name__} must define the 'name' attribute.")

    @abstractmethod
    def forward(self, t: Tensor, /, log_base: Tensor, *, inverse: bool = False) -> Tensor:
        ...
