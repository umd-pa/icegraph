# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import ClassVar

from torch import Tensor

from ...confusion import ConfusionMetric
from ...config import Config

__all__ = ["MacroF1"]


class MacroF1(ConfusionMetric[Config]):
    """Per-head macro-averaged F1 score over segmented predictions.

    Per class, the harmonic mean of precision and recall, averaged with equal
    weight per class. Unlike macro-recall it also penalizes over-firing on a
    class, since false positives enter the denominator. Classes that appear
    neither as a label nor as a prediction are left out of the average.
    """
    name: ClassVar[str] = "macro-f1"
    version: ClassVar[int] = 1

    @property
    def optimum(self) -> float:
        return 1.0

    def repr(self) -> str:
        return "macro_f1"

    def reduce(self, confusion: Tensor) -> Tensor:
        hits = confusion.diagonal().float()     # [K] TP
        true = confusion.sum(dim=1).float()     # [K] TP + FN, row totals
        pred = confusion.sum(dim=0).float()     # [K] TP + FP, column totals

        # 2 TP / (2 TP + FP + FN)
        denominator = true + pred
        f1 = 2.0 * hits / denominator

        return self.macro(f1, denominator > 0)
