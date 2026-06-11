# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import TypeVar
from abc import ABC, abstractmethod

from numpy.typing import ArrayLike

from icegraph.common.plugins import Plugin

from .types import FluxModelContext

__all__ = ["FluxModel"]


C = TypeVar("C")


class FluxModel(Plugin[C, FluxModelContext], ABC):
    """Flux model compatible with simweights weighting."""

    @abstractmethod
    def __call__(self, **kwargs: ArrayLike) -> ArrayLike:
        ...