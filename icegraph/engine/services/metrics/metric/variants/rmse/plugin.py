# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import TypeAlias, Any, ClassVar

import torch
from torch import Tensor

from icegraph.common.tensors import SegmentedTensor

from ...metric import Metric

from .config import RMSEConfig

__all__ = ["RMSE"]


RMSEState: TypeAlias = "tuple[Tensor, Tensor] | None"


class RMSE(Metric[RMSEConfig, RMSEState]):
    """Per-head root mean squared error over segmented predictions.

    The accumulator stays in the *squared* domain — (sum of squared error,
    count) — because the square root does not distribute over addition, so RMSE
    cannot be accumulated or merged incrementally. The root is taken once, in
    ``finalize``, after the full mean is known.
    """
    name: ClassVar[str] = "rmse"
    version: ClassVar[int] = 1

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> RMSEConfig:
        return RMSEConfig(**config)

    @property
    def optimum(self) -> float:
        return 0.0

    def repr(self) -> str:
        return "rmse"

    def initial(self) -> RMSEState:
        return None

    def update_state(
        self, state: RMSEState, out: SegmentedTensor, target: SegmentedTensor
    ) -> RMSEState:
        # both are already on accelerator
        ids     = out.kernel_view.ids
        widths  = out.kernel_view.widths

        o = out.data     # [B, V]
        t = target.data  # [B, V]

        diff = o - t     # out - target
        diff.mul_(diff)  # (out - target) ** 2
        col = diff.sum(dim=0, dtype=torch.float32)

        sse_batch = col.new_zeros(widths.numel()).scatter_add_(0, ids, col)
        cnt_batch = widths * o.shape[0]

        if state is None:
            return sse_batch, cnt_batch

        sse, cnt = state
        sse += sse_batch
        cnt += cnt_batch
        return sse, cnt

    def combine(self, a: RMSEState, b: RMSEState) -> RMSEState:
        # trivial cases
        if a is None:
            return b
        if b is None:
            return a

        sa, ca = a
        sb, cb = b
        return sa + sb, ca + cb

    def finalize(self, state: RMSEState) -> Tensor:
        if state is None:
            # no batches seen yet
            return torch.empty(0)

        sse, cnt = state
        return (sse / cnt).sqrt_()