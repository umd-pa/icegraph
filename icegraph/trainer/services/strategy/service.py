# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from ..service import Service

from .strategy import Strategy
from .factory import StrategyFactory
from .view import StrategyView
from .config import StrategyConfig
from .types import CompatibleModule

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["StrategyService"]


class StrategyService(Service[StrategyView, StrategyConfig]):
    name: ClassVar[str] = "strategy"

    interface = StrategyView

    # make the type checker happy
    _strategy:      Strategy | None
    _in_channels:   int | None
    _out_channels:  int | None

    def build(self) -> None:
        # module initialized on attach
        self._strategy = None

        # cache for in and out channels
        self._in_channels = None
        self._out_channels = None

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> StrategyConfig:
        return StrategyConfig(**config)

    def on_attach(self) -> None:
        # initialize strategy module for given mode (strategies are named by the mode they manage)
        self._strategy = StrategyFactory.create(self.config.name, ctx=self._ctx)

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

    def ensure_compatible(self, module: CompatibleModule) -> None:
        if not module.compatible:
            # an empty compatibility tuple indicates compatibility with any strategy
            return

        if self.mode not in module.compatible:
            raise RuntimeError(f"Module '{type(module).__name__}' is not compatible with strategy mode '{self.mode}'.")

    def state_dict(self) -> dict[str, Any]:
        return {"config": self.config.model_dump(mode="json")}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.config = type(self).validate_config(state["config"])
