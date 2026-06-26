# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import ClassVar, Any

import torch
from torch import Tensor
import torch.nn.functional as F

from icegraph.common.tensors import SegmentedTensor
from icegraph.engine.components.loss import LossFunction

from .config import NLLConfig

__all__ = ["NLLLoss"]


class NLLLoss(LossFunction[NLLConfig]):
    name: ClassVar[str] = "nll"
    version: ClassVar[int] = 1

    compatible = ("multiclass",)

    def build(self) -> None:
        return

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> NLLConfig:
        return NLLConfig(**config)

    def loss(self, out: SegmentedTensor, target: SegmentedTensor, /) -> Tensor:
        # load weights from config
        weight = torch.tensor(
            self.config.weight, device=self.device, dtype=torch.float32
        ) if self.config.weight is not None else None

        # compute loss for each head and reduce
        loss_fn = F.cross_entropy if self.config.from_logits else F.nll_loss

        losses = [
            loss_fn(
                o, t.squeeze(-1),
                reduction=self.config.reduction,
                weight=weight,
                ignore_index=self.config.ignore_index,
            )
            for o, t in zip(out, target, strict=True)
        ]

        # stack and return sum
        return torch.stack(losses).sum()