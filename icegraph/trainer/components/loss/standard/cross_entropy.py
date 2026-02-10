# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import ClassVar, Any

import torch
from torch import Tensor
import torch.nn.functional as F

from ..loss import LossFunction
from ..config import CrossEntropyLossConfig


class CrossEntropyLoss(LossFunction[CrossEntropyLossConfig]):
    name: ClassVar[str] = "cross_entropy"

    compatible = ("multiclass",)

    def build(self) -> None:
        return

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> CrossEntropyLossConfig:
        return CrossEntropyLossConfig(**config)

    def forward(self, out: Tensor, target: Tensor, /) -> Tensor:
        weight = torch.tensor(
            self.config.weight, device=out.device, dtype=torch.float32
        ) if self.config.weight is not None else None

        return F.cross_entropy(
            out, target,
            reduction=self.config.reduction,
            weight=weight,
            ignore_index=self.config.ignore_index,
            label_smoothing=self.config.label_smoothing
        )
