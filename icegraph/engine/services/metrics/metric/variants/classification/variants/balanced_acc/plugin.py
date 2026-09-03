# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Any, ClassVar

from torch import Tensor

from ...confusion import ConfusionMetric

from .config import BalancedAccuracyConfig

__all__ = ["BalancedAccuracy"]


class BalancedAccuracy(ConfusionMetric[BalancedAccuracyConfig]):
    """Per-head balanced accuracy over segmented predictions.

    Accuracy as it would read if every class were equally represented, which is
    the mean of the per-class recalls and so the same quantity as macro-recall.
    With ``adjusted``, rescaled so chance sits at ``0`` instead of ``1 / K``.
    """
    name: ClassVar[str] = "balanced-acc"
    version: ClassVar[int] = 1

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> BalancedAccuracyConfig:
        return BalancedAccuracyConfig(**config)

    @property
    def optimum(self) -> float:
        return 1.0

    def repr(self) -> str:
        return "adj_balanced_acc" if self.config.adjusted else "balanced_acc"

    def reduce(self, confusion: Tensor) -> Tensor:
        rec, support = self.recall(confusion)  # [K], [K]

        present = support > 0
        score   = self.macro(rec, present)

        if not self.config.adjusted:
            return score

        # chance level over the classes actually used, not the declared ones
        chance = 1.0 / present.sum()
        return (score - chance) / (1.0 - chance)
