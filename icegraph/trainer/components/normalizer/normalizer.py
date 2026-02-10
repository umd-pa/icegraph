# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import TypeVar, Generic
from abc import ABC, abstractmethod

import torch
from torch import Tensor

from icegraph.types.data import ModelInputRole

from ..component import Component

from .types import NormalizerContext

__all__ = ["Normalizer"]


C = TypeVar("C")


class Normalizer(Component[C, NormalizerContext], ABC, Generic[C]):

    @abstractmethod
    @torch.no_grad()
    def forward(self, t: Tensor, /, role: ModelInputRole, *, inverse: bool = False) -> Tensor:
        """Forward pass through the normalizer."""
        ...
