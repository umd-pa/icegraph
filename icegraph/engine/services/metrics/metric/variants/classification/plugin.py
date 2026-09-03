# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import TypeVar, Any, Iterator
from abc import ABC

from torch import Tensor

from icegraph.common.tensors import SegmentedTensor

from ...metric import Metric

from .config import Config

__all__ = ["ClassificationMetric"]


C = TypeVar("C")  # plugin config type
S = TypeVar("S")  # accumulator state chosen freely by each metric


class ClassificationMetric(Metric[C, S], ABC):
    """
    Base class for metrics over categorical targets.

    A classification metric reads each head's columns as scores over that head's
    classes and its single target column as a class index, the layout the
    multiclass policy produces. Heads are ragged so the family walks them
    one at a time rather than working across the packed
    row, and :meth:`heads` is the shared entry point that pairs a head's scores
    with its labels and rejects a target block that is not one column per head.

    The family fixes no accumulator: metrics here range from a confusion matrix
    to a binned score histogram. What they do share is the reading of the model
    output, which splits them in two. Anything downstream of the arg-max is
    invariant to whether the output is logits, log-probabilities or probabilities
    and needs no configuration. Anything that reads the distribution itself takes
    a ``from_logits`` option and normalizes through :meth:`probabilities`.
    """

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> Any:
        return Config(**config)

    @staticmethod
    def heads(
        out: SegmentedTensor, target: SegmentedTensor
    ) -> Iterator[tuple[int, Tensor, Tensor]]:
        for head, (o, t) in enumerate(zip(out, target, strict=True)):
            if t.shape[1] != 1:
                raise ValueError(
                    f"Classification metrics expect a single target column per head; "
                    f"head {head} has {t.shape[1]}. Is this a classification run?"
                )

            yield head, o.detach(), t.detach().reshape(-1).long()

    @staticmethod
    def probabilities(scores: Tensor, *, from_logits: bool) -> Tensor:
        scores = scores.float()
        return scores.softmax(dim=1) if from_logits else scores.exp()

    @staticmethod
    def macro(values: Tensor, mask: Tensor) -> Tensor:
        """Mean of ``values`` over the selected entries, as a 1-element tensor."""
        total = values.masked_fill(~mask, 0.0).sum()  # nan slots are masked out first
        return (total / mask.sum()).reshape(1)
