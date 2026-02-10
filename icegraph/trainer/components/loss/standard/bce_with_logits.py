# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import ClassVar, Any

import torch
from torch import Tensor
import torch.nn.functional as F

from ..loss import LossFunction
from ..config import BCEWithLogitsLossConfig


class BCEWithLogitsLoss(LossFunction[BCEWithLogitsLossConfig]):
    name: ClassVar[str] = "bce_with_logits"

    compatible = ("class", "multilabel")

    def build(self) -> None:
        return

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> BCEWithLogitsLossConfig:
        return BCEWithLogitsLossConfig(**config)

    def forward(self, out: Tensor, target: Tensor, /) -> Tensor:
        weight = torch.tensor(
            self.config.weight, device=out.device, dtype=torch.float32
        ) if self.config.weight is not None else None

        pos_weight = torch.tensor(
            self.config.pos_weight, device=out.device, dtype=torch.float32
        ) if self.config.pos_weight is not None else None

        return F.binary_cross_entropy_with_logits(
            out, target,
            reduction=self.config.reduction,
            weight=weight,
            pos_weight=pos_weight
        )
