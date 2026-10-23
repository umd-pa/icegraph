# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import TypeAlias, Any, ClassVar

import torch
from torch import Tensor

from icegraph.common.tensors import SegmentedTensor
from icegraph.engine.services.metrics.types import HeadValues

from ...plugin import ClassificationMetric

from .config import ECEConfig

__all__ = ["ExpectedCalibrationError"]


ECEState: TypeAlias = "tuple[Tensor, Tensor, Tensor] | None"


class ExpectedCalibrationError(ClassificationMetric[ECEConfig, ECEState]):
    """Per-head expected calibration error over segmented predictions.

    Samples are binned by the probability of their predicted class, and the score
    is the population-weighted mean gap between each bin's accuracy and its mean
    confidence (ie whether a stated confidence means what it says). This is the
    standard top-label ECE over ``bins`` equal-width bins on ``[0, 1]``.
    """
    name: ClassVar[str] = "ece"
    version: ClassVar[int] = 1

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> ECEConfig:
        return ECEConfig(**config)

    @property
    def optimum(self) -> float:
        return 0.0

    def repr(self) -> str:
        return f"ece{self.config.bins}"

    def initial(self) -> ECEState:
        return None

    def update_state(
        self, state: ECEState, out: SegmentedTensor, target: SegmentedTensor
    ) -> ECEState:
        # both are already on accelerator
        m = self.config.bins

        population: list[Tensor] = []
        confidence: list[Tensor] = []
        hits:       list[Tensor] = []

        for _, scores, true in self.heads(out, target):
            p = self.probabilities(scores, from_logits=self.config.from_logits)  # [B, K]

            conf, pred = p.max(dim=1)  # [B] top probability and its class

            # equal-width bins on [0, 1]
            # a confidence of exactly 1 belongs to the
            # last bin rather than falling off the end
            index = (conf * m).long().clamp_(max=m - 1)  # [B]

            population.append(conf.new_zeros(m).scatter_add_(0, index, torch.ones_like(conf)))
            confidence.append(conf.new_zeros(m).scatter_add_(0, index, conf))
            hits.append(conf.new_zeros(m).scatter_add_(0, index, (pred == true).float()))

        batch = (torch.stack(population), torch.stack(confidence), torch.stack(hits))  # 3x [L, M]

        if state is None:
            return batch

        return tuple(s + b for s, b in zip(state, batch, strict=True))

    def combine(self, a: ECEState, b: ECEState) -> ECEState:
        # trivial cases
        if a is None:
            return b
        if b is None:
            return a

        return tuple(x + y for x, y in zip(a, b, strict=True))

    def finalize(self, state: ECEState) -> HeadValues:
        if state is None:
            # no batches seen yet
            return ()

        population, confidence, hits = state

        # empty bins carry no weight, so clamping the divisor only avoids a nan
        # that would then be multiplied by zero
        occupied = population.clamp(min=1.0)
        gap = (hits / occupied - confidence / occupied).abs()  # [L, M]

        ece = (population * gap).sum(dim=1) / population.sum(dim=1)  # [L]
        return tuple(e.reshape(1) for e in ece.unbind(0))
