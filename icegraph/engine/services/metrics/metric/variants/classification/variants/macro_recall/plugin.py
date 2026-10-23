# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import ClassVar

from torch import Tensor

from ...confusion import ConfusionMetric
from ...config import Config

__all__ = ["MacroRecall"]


class MacroRecall(ConfusionMetric[Config]):
    """Per-head macro-averaged recall over segmented predictions.

    Recall per class, averaged with equal weight per class, so an imbalanced
    evaluation set cannot let a dominant class carry the score. Classes with no
    support are left out of the average rather than counted as zero.
    """
    name: ClassVar[str] = "macro-recall"
    version: ClassVar[int] = 1

    @property
    def optimum(self) -> float:
        return 1.0

    def repr(self) -> str:
        return "macro_recall"

    def reduce(self, confusion: Tensor) -> Tensor:
        rec, support = self.recall(confusion)  # [K], [K]
        return self.macro(rec, support > 0)
