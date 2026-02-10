# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

from ..types import ServiceContext

if TYPE_CHECKING:
    from torch import Tensor

    from ..data import DataView

__all__ = ["Strategy"]


class Strategy(ABC):
    """
    Base class for task-specific training strategies.

    A Strategy defines how a model computes loss, adapts targets,
    and determines input/output channels.
    """
    name: ClassVar[str]

    def __init__(self, ctx: ServiceContext) -> None:
        self._in: int | None = None
        self._out: int | None = None

        # this module needs direct access to the parent attach context
        self._ctx: ServiceContext = ctx

    @property
    def _data(self) -> DataView:
        return self._ctx.services.require("data", required_by=type(self))

    @property
    @abstractmethod
    def reduction(self) -> str:
        ...

    @property
    def in_channels(self):
        if self._in is None:
            self._in = self._in_channels(self._data)
        return self._in

    @property
    def out_channels(self):
        if self._out is None:
            self._out = self._out_channels(self._data)
        return self._out

    @abstractmethod
    def _out_channels(self, data: DataView) -> int:
        """Return the number of output channels for the model."""
        ...

    @abstractmethod
    def _in_channels(self, data: DataView) -> int:
        """Return the number of input channels for the model."""
        ...

    @abstractmethod
    def adapt_targets(self, targets: Tensor) -> Tensor:
        """
        Adapt raw batch targets to align with strategy.

        Args:
            targets: Tensor of targets.
        """
        ...
