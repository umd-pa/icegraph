# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from abc import abstractmethod

import torch
from torch import Tensor

# local package
from icegraph.common.histogram import Histogram
from icegraph.trainer.callbacks.base.accumulator import Accumulator

# local subpackage
from .base import HistogramReducer


__all__ = ["CHistogramReducer"]


class CHistogramReducer(HistogramReducer):

    def _encode(self, data: Tensor, label: str) -> Tensor:
        # ensure correct dim
        self.ensure_shape(data)

        # no need to floor here, data is already indices
        indices = data.to(torch.int64)

        # build mask
        mask = ((indices >= 0) & (indices < self.bins.on(data.device))).all(dim=-1)

        # fail fast if any indices are out of bounds
        # categorical histograms cannot fundamentally have out-of-bound indices;
        # if they do, that is a major problem
        if not mask.all().item():
            raise ValueError(
                f"Out-of-bound indices detected in {type(self).__name__}. "
                f"This likely indicates invalid class predictions from the model "
                f"or a fault in runtime data processing."
            )

        # flatten indices
        flat = self._flatten(indices)

        # return dense histogram
        return self._to_dense(flat)

    def _build_artifact(self, accumulator: Accumulator, label: str) -> Histogram:
        # build histogram object
        return Histogram(
            space=self.scale,
            histogram=accumulator.data.cpu().numpy()
        )

    @abstractmethod
    def _build_bins(self) -> Tensor:
        ...
