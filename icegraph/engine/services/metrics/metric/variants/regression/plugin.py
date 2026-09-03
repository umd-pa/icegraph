# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import TypeAlias, TypeVar, Any
from abc import abstractmethod, ABC

import torch
from torch import Tensor

from icegraph.common.tensors import SegmentedTensor

from ...metric import Metric, HeadValues

from .config import Config

__all__ = ["RegressionMetric", "RegressionState"]


C = TypeVar("C")  # plugin config type

RegressionState: TypeAlias = "tuple[Tensor, Tensor] | None"


class RegressionMetric(Metric[C, RegressionState], ABC):
    """
    Base class for error metrics over continuous targets.

    Every metric in this family is a mean over elementwise residuals, so they all
    share one accumulator. Both are exactly
    additive, which makes the monoid trivial and the epoch value exact however
    batches happen to divide.

    A subclass supplies the residual it accumulates and, when the reported value
    is not that mean itself, the transform applied to it once at the end. That
    split matters: a transform that does not distribute over addition must be
    deferred to ``resolve`` rather than folded into the
    accumulator, or batching would change the answer.

    Predictions and targets are compared columnwise across the whole packed row,
    then scattered into per-head totals, so heads of any width cost one pass.
    """

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> Any:
        return Config(**config)

    @property
    def optimum(self) -> float:
        # an error metric is best at zero
        return 0.0

    def initial(self) -> RegressionState:
        return None

    def update_state(
        self, state: RegressionState, out: SegmentedTensor, target: SegmentedTensor
    ) -> RegressionState:
        # both are already on accelerator
        ids     = out.ids
        widths  = out.widths

        diff = out.data - target.data                                   # [B, V]
        col  = self.residual(diff).sum(dim=0, dtype=torch.float32)      # [V]

        total_batch = col.new_zeros(widths.numel()).scatter_add_(0, ids, col)
        count_batch = widths * out.data.shape[0]

        if state is None:
            return total_batch, count_batch

        total, count = state
        total += total_batch
        count += count_batch
        return total, count

    def combine(self, a: RegressionState, b: RegressionState) -> RegressionState:
        # trivial cases
        if a is None:
            return b
        if b is None:
            return a

        ta, ca = a
        tb, cb = b
        return ta + tb, ca + cb

    def finalize(self, state: RegressionState) -> HeadValues:
        if state is None:
            # no batches seen yet
            return ()

        total, count = state
        value = self.resolve(total / count)  # [L]
        return tuple(v.reshape(1) for v in value.unbind(0))

    @abstractmethod
    def residual(self, diff: Tensor) -> Tensor:
        """Elementwise error to accumulate, given ``out - target``.

        ``diff`` is a fresh tensor, so it may be rewritten in place.
        """
        ...

    def resolve(self, mean: Tensor) -> Tensor:
        """Map the per-head mean residual to the reported value."""
        return mean
