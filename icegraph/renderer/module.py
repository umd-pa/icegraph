# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Generic, TypeVar, TYPE_CHECKING, ClassVar
from abc import ABC, abstractmethod

import plotly.graph_objects as go

if TYPE_CHECKING:
    from .plotter import Plotter

__all__ = ["PlotterModule"]


T = TypeVar("T")


class PlotterModule(ABC, Generic[T]):
    """Base class for IceGraph plot modules."""

    compatible: ClassVar[tuple[type[Plotter], ...]] = tuple()

    @abstractmethod
    def overlay(self, fig: go.Figure, data: T) -> None:
        ...

    def validate_for(self, plotter: Plotter) -> None:
        if not isinstance(plotter, self.compatible):
            raise TypeError(
                f"{type(self).__name__} is not compatible with {type(plotter).__name__}. "
                f"Compatible types: {[c.__name__ for c in self.compatible]}"
            )

