# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import ClassVar

from torch import Tensor

from ...confusion import ConfusionMetric
from ...config import Config

__all__ = ["PerClassRecall"]


class PerClassRecall(ConfusionMetric[Config]):
    """Per-head, per-class recall over segmented predictions.

    Reports one value per class rather than one per head, so a head's slot is a
    vector as wide as its class count. A class with no support is ``nan``.
    """
    name: ClassVar[str] = "per-class-recall"
    version: ClassVar[int] = 1

    @property
    def optimum(self) -> float:
        return 1.0

    def repr(self) -> str:
        return "per_class_recall"

    def reduce(self, confusion: Tensor) -> Tensor:
        rec, _ = self.recall(confusion)  # [K]
        return rec
