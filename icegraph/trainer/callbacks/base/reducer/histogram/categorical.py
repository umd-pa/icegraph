# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

import torch
from torch import Tensor

from icegraph.common.histogram import Histogram

# local subpackage
from .base import HistogramReducer


__all__ = ["CHistogramReducer"]


class CHistogramReducer(HistogramReducer):
    """Histogram reducer over rows that are already per-axis class indices."""

    def encode(self, rows: Tensor, label: str) -> Tensor:
        bins = self.bins[label][rows.device]

        # no need to floor here, rows are already indices
        indices = rows.to(torch.int64)

        # fail fast on out-of-bound indices; categorical data cannot fundamentally
        # have them, so if it does that is a major problem
        if not ((indices >= 0) & (indices < bins)).all().item():
            raise ValueError(
                f"Out-of-bound indices detected in {type(self).__name__}. "
                f"This likely indicates invalid class predictions from the model "
                f"or a fault in runtime data processing."
            )

        return self._count(indices, label)

    def histogram(self, t: Tensor, label: str) -> Histogram:
        return Histogram(histogram=t.cpu().numpy())

