# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import ClassVar, Any

import torch
from torch import Tensor
import torch.nn.functional as F

from icegraph.trainer.components.loss import LossFunction

from .config import Config

__all__ = ["CrossEntropyLoss"]


class CrossEntropyLoss(LossFunction[Config]):
    name: ClassVar[str] = "cross_entropy"
    version: ClassVar[int] = 1

    compatible = ("multiclass",)

    def build(self) -> None:
        return

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> Config:
        return Config(**config)

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
