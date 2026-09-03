# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import TypeAlias, TypeVar
from abc import abstractmethod, ABC

import torch
from torch import Tensor

from icegraph.common.tensors import SegmentedTensor

from ...metric import HeadValues

from .plugin import ClassificationMetric

__all__ = ["ConfusionMetric", "ConfusionState"]


C = TypeVar("C")  # plugin config type


ConfusionState: TypeAlias = "tuple[Tensor, ...] | None"


class ConfusionMetric(ClassificationMetric[C, ConfusionState], ABC):
    """
    Base class for classification metrics that are a function of the confusion matrix.

    The accumulator holds one ``[K, K]`` matrix per head, indexed
    ``[true, predicted]``, which is exactly additive: counting over two batches
    and adding is counting over their union. That makes the monoid trivial and
    leaves subclasses to implement only ``reduce``, the pure function from one
    head's matrix to that head's value.

    Predictions are the arg-max over a head's columns, so the matrix is the same
    whether the model emits logits, log-probabilities or probabilities.
    """

    def initial(self) -> ConfusionState:
        return None

    def update_state(
        self, state: ConfusionState, out: SegmentedTensor, target: SegmentedTensor
    ) -> ConfusionState:
        # both are already on accelerator
        batch: list[Tensor] = []

        for _, scores, true in self.heads(out, target):
            k = scores.shape[1]  # class count of this head

            pred = scores.argmax(dim=1)  # [B]

            # one bin per (true, predicted) cell, counted flat then folded
            flat  = true * k + pred      # [B]
            cells = flat.new_zeros(k * k).scatter_add_(0, flat, torch.ones_like(flat))

            batch.append(cells.view(k, k))

        if state is None:
            return tuple(batch)

        return tuple(m + b for m, b in zip(state, batch, strict=True))

    def combine(self, a: ConfusionState, b: ConfusionState) -> ConfusionState:
        # trivial cases
        if a is None:
            return b
        if b is None:
            return a

        return tuple(x + y for x, y in zip(a, b, strict=True))

    def finalize(self, state: ConfusionState) -> HeadValues:
        if state is None:
            # no batches seen yet
            return ()

        return tuple(self.reduce(confusion) for confusion in state)

    @staticmethod
    def recall(confusion: Tensor) -> tuple[Tensor, Tensor]:
        """Per-class recall and support for one head's ``[K, K]`` confusion matrix.

        Rows are true classes, so the row sums are the support. A class with no
        support has undefined recall and comes back as ``nan``. The support vector
        is returned alongside so callers can mask those classes out.
        """
        support = confusion.sum(dim=1)                  # [K] true counts
        hits    = confusion.diagonal()                  # [K]

        return hits.float() / support.float(), support  # 0 / 0 -> nan

    @abstractmethod
    def reduce(self, confusion: Tensor) -> Tensor:
        """Resolve one head's ``[K, K]`` confusion matrix to a 1-D value."""
        ...
