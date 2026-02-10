# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import ClassVar, Any

from torch import Tensor
import torch.nn.functional as F

from ..loss import LossFunction
from ..config import L1LossConfig

__all__ = ["L1Loss"]


class L1Loss(LossFunction[L1LossConfig]):
    name: ClassVar[str] = "l1"

    compatible = ("regression",)

    def build(self) -> None:
        return

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> L1LossConfig:
        return L1LossConfig(**config)

    def forward(self, out: Tensor, target: Tensor, /) -> Tensor:
        return F.l1_loss(out, target, reduction=self.config.reduction)
