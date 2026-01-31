# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import TYPE_CHECKING

from icegraph.types.data import ModelInputRole

from ..strategy import Strategy

if TYPE_CHECKING:
    from torch import Tensor

    from icegraph.trainer.services.data import DataView

__all__ = ["Regression"]


class Regression(Strategy):
    name = "regression"

    @property
    def reduction(self) -> str:
        return "mean"

    def _out_channels(self, data: DataView) -> int:
        return len(data.global_attrs.columns(ModelInputRole.LABELS))

    def _in_channels(self, data: DataView) -> int:
        return len(data.global_attrs.columns(ModelInputRole.FEATURES))

    def adapt_targets(self, targets: Tensor) -> Tensor:
        # [y] -> [y, 1]
        if targets.dim() == 1:
            targets = targets.unsqueeze(1)

        # cast to fp32 if required
        return targets.float()
