# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Any, ClassVar

import torch
from torch import Tensor

from ...confusion import ConfusionMetric

from .config import CohenKappaConfig

__all__ = ["CohenKappa"]


class CohenKappa(ConfusionMetric[CohenKappaConfig]):
    """Per-head Cohen's kappa over segmented predictions.

    Agreement between prediction and truth scored against what two independent
    raters with the same marginals would reach by chance, so a model that has
    only learned the class prior scores around ``0`` however high its raw
    accuracy. ``weights`` generalizes this to ordered classes.
    """
    name: ClassVar[str] = "cohen-kappa"
    version: ClassVar[int] = 1

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> CohenKappaConfig:
        return CohenKappaConfig(**config)

    @property
    def optimum(self) -> float:
        return 1.0

    def repr(self) -> str:
        weights = self.config.weights
        return "cohen_kappa" if weights == "none" else f"cohen_kappa_{weights}"

    def _weights(self, k: int, *, device: torch.device) -> Tensor:
        """Disagreement cost of predicting ``j`` when the truth is ``i``, as ``[K, K]``."""
        index = torch.arange(k, device=device, dtype=torch.float32)
        gap   = index.unsqueeze(1) - index.unsqueeze(0)  # [K, K] i - j

        if self.config.weights == "linear":
            return gap.abs()
        if self.config.weights == "quadratic":
            return gap * gap

        # every off-diagonal cell costs the same
        return 1.0 - torch.eye(k, device=device, dtype=torch.float32)

    def reduce(self, confusion: Tensor) -> Tensor:
        observed = confusion.float()
        total    = observed.sum()

        # chance model keeps marginals and assumes independence
        true = observed.sum(dim=1)  # [K]
        pred = observed.sum(dim=0)  # [K]
        expected = torch.outer(true, pred) / total

        # 1 - (weighted disagreement observed) / (weighted disagreement by chance)
        # the shared 1 / N cancels between the two sums, so raw counts are fine
        weights = self._weights(observed.shape[0], device=observed.device)
        kappa = 1.0 - (weights * observed).sum() / (weights * expected).sum()

        return kappa.reshape(1)
