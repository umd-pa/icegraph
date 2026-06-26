# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Self

import torch
from torch import Tensor

__all__ = ["Accumulator"]


@dataclass
class Accumulator(ABC):

    @abstractmethod
    def update(self, data: Tensor, /) -> None:
        ...

    @property
    @abstractmethod
    def data(self) -> Tensor:
        ...

    @data.setter
    @abstractmethod
    def data(self, data: Tensor) -> None:
        ...

    @abstractmethod
    def reset(self) -> None:
        ...

    @abstractmethod
    def is_empty(self) -> bool:
        ...

    @abstractmethod
    def to(self, device: torch.device | str) -> Self:
        ...
