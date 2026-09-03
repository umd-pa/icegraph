# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import TypeAlias, Any, ClassVar

import torch
from torch import Tensor

from icegraph.common.tensors import SegmentedTensor
from icegraph.engine.services.metrics.types import HeadValues

from ...plugin import ClassificationMetric

from .config import TopKAccuracyConfig

__all__ = ["TopKAccuracy"]


# additive state: (correct [L], count [L]); None is the monoid identity.
TopKAccuracyState: TypeAlias = "tuple[Tensor, Tensor] | None"


class TopKAccuracy(ClassificationMetric[TopKAccuracyConfig, TopKAccuracyState]):
    """
    Per-head top-k classification accuracy over segmented predictions.

    A sample is top-k correct for a head iff fewer than ``k`` classes in that head
    strictly outscore the true class (true class ranks ≤ k). This sidesteps a real
    top-k / sort over ragged segments: it's a segmented count of
    "score > true-class score", computed as a scatter-add.

    That count runs across the whole packed row at once, so unlike the rest of the
    family this metric does not walk heads one at a time.
    """
    name: ClassVar[str] = "top-k-acc"
    version: ClassVar[int] = 1

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> TopKAccuracyConfig:
        return TopKAccuracyConfig(**config)

    @property
    def optimum(self) -> float:
        return 1.0

    def repr(self) -> str:
        return f"top{self.config.k}_acc"

    def initial(self) -> TopKAccuracyState:
        return None

    def update_state(
        self, state: TopKAccuracyState, out: SegmentedTensor, target: SegmentedTensor
    ) -> TopKAccuracyState:
        # both are already on accelerator
        ids     = out.ids        # [V]
        widths  = out.widths     # [L]
        L       = widths.numel()

        o      = out.data.detach()           # [B, V] class scores
        labels = target.data.detach()        # [B, L] local class index per head
        B      = o.shape[0]
        k      = self.config.k

        # the family invariant, checked here since the packed path never walks heads
        if labels.shape[1] != L:
            raise ValueError(
                f"Classification metrics expect a single target column per head; "
                f"got {labels.shape[1]} columns for {L} heads. Is this a classification run?"
            )

        # global column of each heads true class
        starts = widths.cumsum(0) - widths               # [L]
        gidx   = (starts.unsqueeze(0) + labels).long()   # [B, L] global true-class columns
        s_star = o.gather(1, gidx)                        # [B, L] score of the true class

        # broadcast true score across each heads columns, count strictly better
        idx        = ids.unsqueeze(0).expand(B, -1)       # [B, V] per row scatter index
        s_star_col = s_star[:, ids]                       # [B, V]
        greater    = (o > s_star_col).to(torch.int32)     # [B, V]
        gcount     = greater.new_zeros(B, L).scatter_add_(1, idx, greater)  # [B, L]

        # top-k correct
        correct_batch = (gcount < k).sum(dim=0).to(torch.long)             # [L]
        cnt_batch     = torch.full((L,), B, device=o.device, dtype=torch.long)

        if state is None:
            return correct_batch, cnt_batch

        correct, cnt = state
        correct += correct_batch
        cnt     += cnt_batch
        return correct, cnt

    def combine(
        self, a: TopKAccuracyState, b: TopKAccuracyState
    ) -> TopKAccuracyState:
        # trivial cases
        if a is None:
            return b
        if b is None:
            return a

        ca, na = a
        cb, nb = b
        return ca + cb, na + nb

    def finalize(self, state: TopKAccuracyState) -> HeadValues:
        if state is None:
            # no batches seen yet
            return ()

        correct, cnt = state
        acc = correct / cnt  # [L]
        return tuple(a.reshape(1) for a in acc.unbind(0))
