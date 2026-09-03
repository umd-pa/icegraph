# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import TypeAlias, Any, ClassVar

import torch
from torch import Tensor
import torch.nn.functional as F

from icegraph.common.tensors import SegmentedTensor
from icegraph.engine.services.metrics.types import HeadValues

from ...plugin import ClassificationMetric

from .config import CrossEntropyConfig

__all__ = ["CrossEntropy"]


CrossEntropyState: TypeAlias = "tuple[Tensor, Tensor] | None"


class CrossEntropy(ClassificationMetric[CrossEntropyConfig, CrossEntropyState]):
    """Per-head cross-entropy over segmented predictions, in nats.

    The mean negative log-probability assigned to the true class. It reads the
    whole predicted distribution rather than its arg-max, so it separates a model
    that is right by a hair from one that is right with conviction. The
    accumulator holds the summed likelihood, so the mean is exact however batches
    divide.
    """
    name: ClassVar[str] = "cross-entropy"
    version: ClassVar[int] = 1

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> CrossEntropyConfig:
        return CrossEntropyConfig(**config)

    @property
    def optimum(self) -> float:
        return 0.0

    def repr(self) -> str:
        return "cross_entropy"

    def initial(self) -> CrossEntropyState:
        return None

    def update_state(
        self, state: CrossEntropyState, out: SegmentedTensor, target: SegmentedTensor
    ) -> CrossEntropyState:
        # both are already on accelerator
        loss_fn = F.cross_entropy if self.config.from_logits else F.nll_loss

        # summed rather than averaged
        # a per-batch mean could not be folded into a running total without reweighting
        nll = torch.stack([
            loss_fn(scores.float(), true, reduction="sum")
            for _, scores, true in self.heads(out, target)
        ])  # [L]

        cnt_batch = torch.full_like(nll, out.data.shape[0])

        if state is None:
            return nll, cnt_batch

        nll_sum, cnt = state
        nll_sum += nll
        cnt     += cnt_batch
        return nll_sum, cnt

    def combine(self, a: CrossEntropyState, b: CrossEntropyState) -> CrossEntropyState:
        # trivial cases
        if a is None:
            return b
        if b is None:
            return a

        sa, ca = a
        sb, cb = b
        return sa + sb, ca + cb

    def finalize(self, state: CrossEntropyState) -> HeadValues:
        if state is None:
            # no batches seen yet
            return ()

        nll_sum, cnt = state
        ce = nll_sum / cnt  # [L]
        return tuple(c.reshape(1) for c in ce.unbind(0))
