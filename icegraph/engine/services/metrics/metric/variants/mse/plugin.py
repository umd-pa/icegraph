# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import TypeAlias, Any, ClassVar

import torch
from torch import Tensor

from icegraph.common.tensors import SegmentedTensor

from ...metric import Metric

from .config import MSEConfig

__all__ = ["MSE"]


MSEState: TypeAlias = "tuple[Tensor, Tensor] | None"


class MSE(Metric[MSEConfig, MSEState]):
    """Per-head mean squared error over segmented predictions."""
    name: ClassVar[str] = "mse"
    version: ClassVar[int] = 1

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> MSEConfig:
        return MSEConfig(**config)

    @property
    def optimum(self) -> float:
        return 0.0

    def repr(self) -> str:
        return "mse"

    def initial(self) -> MSEState:
        return None

    def update_state(
        self, state: MSEState, out: SegmentedTensor, target: SegmentedTensor
    ) -> MSEState:
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

    def combine(self, a: MSEState, b: MSEState) -> MSEState:
        # trivial cases
        if a is None:
            return b
        if b is None:
            return a

        sa, ca = a
        sb, cb = b
        return sa + sb, ca + cb

    def finalize(self, state: MSEState) -> Tensor:
        if state is None:
            # no batches seen yet
            return torch.empty(0)

        sse, cnt = state
        return sse / cnt