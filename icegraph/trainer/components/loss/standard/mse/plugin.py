# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import ClassVar, Any

from torch import Tensor
import torch.nn.functional as F

from icegraph.trainer.components.loss import LossFunction

from .config import Config

__all__ = ["MSELoss"]


class MSELoss(LossFunction[Config]):
    name: ClassVar[str] = "mse"
    version: ClassVar[int] = 1

    compatible = ("regression",)

    def build(self) -> None:
        return

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> Config:
        return Config(**config)

    def forward(self, out: Tensor, target: Tensor, /) -> Tensor:
        return F.mse_loss(out, target, reduction=self.config.reduction)