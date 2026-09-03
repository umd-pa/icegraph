# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import TypeAlias, Any, ClassVar

import torch
from torch import Tensor

from icegraph.common.tensors import SegmentedTensor

from ...metric import Metric, HeadValues

from .config import MAEConfig

__all__ = ["MAE"]


MAEState: TypeAlias = "tuple[Tensor, Tensor] | None"


class MAE(Metric[MAEConfig, MAEState]):
    """Per-head mean absolute error over segmented predictions."""
    name: ClassVar[str] = "mae"
    version: ClassVar[int] = 1

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> MAEConfig:
        return MAEConfig(**config)

    @property
    def optimum(self) -> float:
        return 0.0

    def repr(self) -> str:
        return "mae"

    def initial(self) -> MAEState:
        return None

    def update_state(
        self, state: MAEState, out: SegmentedTensor, target: SegmentedTensor
    ) -> MAEState:
        # both are already on accelerator
        ids     = out.ids
        widths  = out.widths

        o = out.data     # [B, V]
        t = target.data  # [B, V]

        diff = o - t     # out - target
        diff.abs_()      # |out - target|
        col = diff.sum(dim=0, dtype=torch.float32)

        sae_batch = col.new_zeros(widths.numel()).scatter_add_(0, ids, col)
        cnt_batch = widths * o.shape[0]

        if state is None:
            return sae_batch, cnt_batch

        sae, cnt = state
        sae += sae_batch
        cnt += cnt_batch
        return sae, cnt

    def combine(self, a: MAEState, b: MAEState) -> MAEState:
        # trivial cases
        if a is None:
            return b
        if b is None:
            return a

        sa, ca = a
        sb, cb = b
        return sa + sb, ca + cb

    def finalize(self, state: MAEState) -> HeadValues:
        if state is None:
            # no batches seen yet
            return ()

        sae, cnt = state
        mae = sae / cnt                                   # [L]
        return tuple(m.reshape(1) for m in mae.unbind(0))