# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import ClassVar, Any

import torch
from torch import Tensor
import torch.nn.functional as F

from icegraph.common.tensors import SegmentedTensor
from icegraph.engine.components.loss import LossFunction

from .config import MSEConfig

__all__ = ["MSELoss"]


class MSELoss(LossFunction[MSEConfig]):
    name: ClassVar[str] = "mse"
    version: ClassVar[int] = 1

    compatible = ("regression",)

    def build(self) -> None:
        return

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> MSEConfig:
        return MSEConfig(**config)

    def loss(self, out: SegmentedTensor, target: SegmentedTensor, /, weights: Tensor) -> Tensor:
        # compute loss for each head and reduce
        losses = []
        for o, t in zip(out, target, strict=True):
            if not self.config.ignore_nan:
                losses.append(F.mse_loss(o, t, reduction=self.config.reduction))
                continue

            keep = ~t.isnan()
            loss = F.mse_loss(o[keep], t[keep], reduction="sum")

            # a head with no target in the batch adds nothing, rather than NaN
            losses.append(loss / keep.sum().clamp(min=1) if self.config.reduction == "mean" else loss)

        # stack and return sum
        return torch.stack(losses).sum()