# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

"""Shared one-vs-rest helpers for threshold-curve reducers (ROC, PR)."""

from __future__ import annotations

from collections.abc import Iterator, Mapping

import torch
from torch import Tensor

__all__ = ["project_ovr", "iter_ovr", "bin_curve"]


def project_ovr(out: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
    """
    Emit one score per (sample, class), grouped one-vs-rest.

    Group ``2c`` holds class ``c``'s scores on negatives, ``2c + 1`` on positives.
    """
    probs = out.softmax(dim=-1)  # [B, C]

    classes = torch.arange(probs.shape[1], device=out.device).unsqueeze(0)  # [1, C]
    groups = 2 * classes + target.eq(classes).long()                        # [B, 1] x [1, C] -> [B, C]

    return probs.reshape(-1, 1), groups.flatten()


def iter_ovr(counts: Mapping[int, Tensor]) -> Iterator[tuple[int, Tensor, Tensor]]:
    """Yield ``(class, neg, pos)`` float score histograms for each class with both sides present."""
    for c in range(max(counts) // 2 + 1):
        neg, pos = counts.get(2 * c), counts.get(2 * c + 1)

        if neg is not None and pos is not None:
            yield c, neg.float().flatten(), pos.float().flatten()


def bin_curve(x: Tensor, y: Tensor, n: int) -> Tensor:
    """Resample a curve over ``x`` in ``[0, 1]`` onto ``n`` bins, keeping the max ``y`` per bin."""
    index = (x * n).long().clamp(0, n - 1)
    return torch.zeros(n, dtype=y.dtype, device=y.device).scatter_reduce(0, index, y, reduce="amax")
