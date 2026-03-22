# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, TypeVar

from icegraph.types.plugins import Plugin

from .types import StrategyContext

if TYPE_CHECKING:
    from torch import Tensor

    from icegraph.trainer.services.data import DataView

__all__ = ["Strategy"]


C = TypeVar("C")


class Strategy(Plugin[C, StrategyContext]):
    """Base class for task-specific training strategies."""

    @property
    def _data(self) -> DataView:
        return self._ctx.data

    @abstractmethod
    def in_channels(self):
        ...

    @abstractmethod
    def out_channels(self):
        ...

    @abstractmethod
    def adapt_targets(self, targets: Tensor) -> Tensor:
        """
        Adapt raw batch targets to align with strategy.

        Args:
            targets: Tensor of targets.
        """
        ...
