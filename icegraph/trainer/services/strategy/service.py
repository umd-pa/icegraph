# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from icegraph.trainer.types import Params

from ..service import Service
from ..types import ServiceContext

from .strategy import Strategy
from .factory import StrategyFactory
from .view import StrategyView

if TYPE_CHECKING:
    from torch import Tensor

    from ..data import DataView

__all__ = ["StrategyService"]


class StrategyService(Service):
    name = "strategy"
    deps = []  # data is not a dependency, strategy only relies on data at runtime post service construction
    view = StrategyView

    def __init__(self, params: Params) -> None:
        super().__init__(params)

        # module initialized on attach
        self._strategy: Strategy | None = None

        # cache for in and out channels
        self._in_channels: int | None = None
        self._out_channels: int | None = None

    def on_attach(self, ctx: ServiceContext) -> None:
        # initialize strategy module for given mode (strategies are named by the mode they manage)
        name: str = self.params.require("name")
        self._strategy: Strategy = StrategyFactory.create(name, ctx=ctx)

    def state_dict(self) -> dict[str, Any]:
        self._strategy = None
        self._ctx = None
        return self.__dict__.copy()

    @property
    def mode(self) -> str:
        return self._strategy.name

    @property
    def in_channels(self) -> int:
        """Return the number of input channels for the model."""
        return self._strategy.in_channels

    @property
    def out_channels(self) -> int:
        """Return the number of output channels for the model."""
        return self._strategy.out_channels

    def adapt_targets(self, targets: Tensor) -> Tensor:
        return self._strategy.adapt_targets(targets)
