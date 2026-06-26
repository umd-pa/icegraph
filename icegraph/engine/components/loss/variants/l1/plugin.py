# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import ClassVar, Any

import torch
from torch import Tensor
import torch.nn.functional as F

from icegraph.common.tensors import SegmentedTensor
from icegraph.engine.components.loss import LossFunction

from .config import L1Config

__all__ = ["L1Loss"]


class L1Loss(LossFunction[L1Config]):
    name: ClassVar[str] = "l1"
    version: ClassVar[int] = 1

    compatible = ("regression",)

    def build(self) -> None:
        return

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> L1Config:
        return L1Config(**config)

    def loss(self, out: SegmentedTensor, target: SegmentedTensor, /) -> Tensor:
        # compute loss for each head and reduce
        losses = [
            F.l1_loss(o, t, reduction=self.config.reduction)
            for o, t in zip(out, target, strict=True)
        ]

        # stack and return sum
        return torch.stack(losses).sum()