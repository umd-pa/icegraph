# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Any

from icegraph.types.data import ModelInputRole

from ...strategy import Strategy

from .config import Config

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["Regression"]


class Regression(Strategy[Config]):
    name: ClassVar[str] = "regression"
    version: ClassVar[int] = 1

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> Config:
        return Config(**config)

    def build(self) -> None:
        return

    def out_channels(self) -> int:
        return len(self._ctx.data.columns(ModelInputRole.LABELS))

    def in_channels(self) -> int:
        return len(self._ctx.data.columns(ModelInputRole.FEATURES))

    def adapt_targets(self, targets: Tensor) -> Tensor:
        # [y] -> [y, 1]
        if targets.dim() == 1:
            targets = targets.unsqueeze(1)

        # cast to fp32 if required
        return targets.float()
