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

from .config import AUPRCConfig

__all__ = ["AUPRC"]


AUPRCState: TypeAlias = "tuple[Tensor, ...] | None"


class AUPRC(ClassificationMetric[AUPRCConfig, AUPRCState]):
    """Per-head macro-averaged area under the precision-recall curve.

    Each class is scored one-vs-rest and its average precision summarizes ranking
    quality across every operating point, not just the one an arg-max implies.
    The per-class areas are then averaged with equal weight.

    An exact curve would need every score held and sorted, so scores instead go
    into ``bins`` equal-width bins kept separately for positives and negatives,
    and the threshold sweep runs over bin edges. The result is exact to bin
    resolution. Classes with no positives are left out of the average.
    """
    name: ClassVar[str] = "auprc"
    version: ClassVar[int] = 1

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> AUPRCConfig:
        return AUPRCConfig(**config)

    @property
    def optimum(self) -> float:
        return 1.0

    def repr(self) -> str:
        return f"auprc{self.config.bins}"

    def initial(self) -> AUPRCState:
        return None

    def update_state(
        self, state: AUPRCState, out: SegmentedTensor, target: SegmentedTensor
    ) -> AUPRCState:
        # both are already on accelerator
        m = self.config.bins
        batch: list[Tensor] = []

        for _, scores, true in self.heads(out, target):
            p = self.probabilities(scores, from_logits=self.config.from_logits)  # [B, K]
            k = p.shape[1]

            classes = torch.arange(k, device=p.device)

            # bin every class score, then flatten (class, bin) into one axis so the
            # whole [B, K] block folds in with a single scatter
            index = (p * m).long().clamp_(max=m - 1)         # [B, K]
            cells = (classes * m + index).reshape(-1)        # [B * K]

            # a column is a positive for exactly the one class that is the label
            positive = (true.unsqueeze(1) == classes).reshape(-1)

            total = cells.new_zeros(k * m).scatter_add_(0, cells, torch.ones_like(cells))
            hits  = cells.new_zeros(k * m).scatter_add_(0, cells, positive.long())

            batch.append(torch.stack([hits, total - hits]).view(2, k, m))

        if state is None:
            return tuple(batch)

        return tuple(h + b for h, b in zip(state, batch, strict=True))

    def combine(self, a: AUPRCState, b: AUPRCState) -> AUPRCState:
        # trivial cases
        if a is None:
            return b
        if b is None:
            return a

        return tuple(x + y for x, y in zip(a, b, strict=True))

    def finalize(self, state: AUPRCState) -> HeadValues:
        if state is None:
            # no batches seen yet
            return ()

        return tuple(self._average_precision(hist) for hist in state)

    def _average_precision(self, histogram: Tensor) -> Tensor:
        """Macro-average the one-vs-rest average precision of a ``[2, K, M]`` histogram."""
        positive, negative = histogram.float().unbind(0)  # 2x [K, M]

        # sweep the threshold downwards
        # bin M - 1 first, so a prefix of the
        # reversed histogram is everything scored at or above the current edge
        tp = positive.flip(-1).cumsum(-1)  # [K, M]
        fp = negative.flip(-1).cumsum(-1)  # [K, M]

        support = positive.sum(dim=-1)  # [K] positives per class

        # both divisors are clamped only where the numerator is zero too
        precision = tp / (tp + fp).clamp(min=1.0)               # [K, M]
        recall    = tp / support.clamp(min=1.0).unsqueeze(-1)   # [K, M]

        # average precision as the step sum over recall gains, a bin no sample
        # landed in gains no recall and drops out on its own
        gain = recall - F.pad(recall[..., :-1], (1, 0))
        return self.macro((gain * precision).sum(dim=-1), support > 0)
