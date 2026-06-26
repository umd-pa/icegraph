# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Callable

from abc import abstractmethod
from functools import cached_property, lru_cache

import torch
from torch import Tensor

# local package
from icegraph.common.histogram import Histogram
from icegraph.common.tensors import DualResidentTensor
from icegraph.common.data import Split, DataRole
from icegraph.trainer.callbacks.base.accumulator import Accumulator
from icegraph.statistics import StatisticService
from icegraph.common.transforms import TransformSpace

# local subpackage
from .base import HistogramReducer
from .types import DualResidentBounds

__all__ = ["BHistogramReducer"]


class BHistogramReducer(HistogramReducer):

    @cached_property
    def transforms(self) -> tuple[Callable[[Tensor], Tensor], ...]:
        options = {
            TransformSpace.LOG: torch.log10,
            TransformSpace.ASINH: torch.asinh,
            TransformSpace.LINEAR: lambda t: t
        }

        transforms = tuple(options[space] for space in self.scale)

        return transforms

    def transform(self, t: Tensor, /) -> Tensor:
        # get transforms
        transforms = self.transforms

        if all(f is transforms[0] for f in transforms):
            return transforms[0](t)

        if t.shape[-1] != len(transforms):
            raise ValueError(
                f"Expected last dimension size {len(transforms)}, got {t.shape[-1]}."
            )

        parts = t.unbind(dim=-1)

        transformed = [
            transform(part)
            for part, transform in zip(parts, transforms, strict=True)
        ]

        return torch.stack(transformed, dim=-1)

    def _apply_margin(self, mins: Tensor, maxs: Tensor) -> tuple[Tensor, Tensor]:
        margin = self._kwargs.get("margin")
        if margin is None:
            return mins, maxs

        span = maxs - mins  # [d]

        # normalize margin
        margin = torch.as_tensor(margin, dtype=span.dtype, device=span.device)

        if margin.numel() != 2 * span.numel():
            raise ValueError(f"expected {2 * span.numel()} margin values")

        margin = margin.view(span.numel(), 2).T  # [2, d]

        mins = mins - span * margin[0]
        maxs = maxs + span * margin[1]

        return mins, maxs

    @cached_property
    def bin_scale(self) -> dict[str, DualResidentTensor]:
        bin_scales: dict[str, DualResidentTensor] = {}

        # build the scale for each label
        for label in self._target_labels:
            mins, maxs = self.bounds(label).on("cpu")

            # build scale using bins / (max - min)
            bin_scales[label] = DualResidentTensor(
                self.bins.on("cpu") / (maxs - mins).clamp_min(1e-12)
            )

        return bin_scales

    def _encode(self, data: Tensor, label: str) -> Tensor:
        # make sure data has correct shape
        self.ensure_shape(data)

        # transform data to space
        data = self.transform(data)

        # grab mins and bin scale for current label
        mins, maxs = self.bounds(label).on(data.device)
        bin_scale = self.bin_scale[label].on(data.device)

        # mask to within bounds, including the upper edge
        mask = ((data >= mins.unsqueeze(0)) & (data <= maxs.unsqueeze(0))).all(dim=-1)
        data = data[mask]

        # continuous -> discrete bin indices
        indices = (data - mins.unsqueeze(0)) * bin_scale.unsqueeze(0)
        indices = torch.floor(indices).to(torch.int64)

        # make right edge inclusive by assigning max values to final bin
        indices = torch.clamp(indices, min=0)
        indices = torch.minimum(indices, self.bins.on(data.device).unsqueeze(0) - 1)

        # flatten indices
        flat = self._flatten(indices)

        # return dense histogram
        return self._to_dense(flat)

    def _build_artifact(self, accumulator: Accumulator, label: str) -> Histogram:
        # stack bounds as expected by the histogram
        bounds = torch.stack(self.bounds(label).on("cpu"), dim=0).numpy()

        # build histogram object
        return Histogram(
            space=self.scale,
            histogram=accumulator.data.cpu().numpy(),
            bounds=bounds
        )

    @lru_cache(maxsize=None)
    def bounds(self, label: str) -> DualResidentBounds:
        # create a copy of stats to pass downstream
        stats = self._ctx.engine.decode.get_stats(Split.TRAIN, DataRole.TARGETS).copy()

        # iterate over each label
        mins, maxs = self._build_bounds(stats, label)

        # transform each according to scaling
        mins = self.transform(mins)
        maxs = self.transform(maxs)

        # ensure min stays min and max stays max
        mins, maxs = torch.minimum(mins, maxs), torch.maximum(mins, maxs)

        # apply margin factor
        mins, maxs = self._apply_margin(mins, maxs)

        return DualResidentBounds(
            mins=DualResidentTensor(mins),
            maxs=DualResidentTensor(maxs),
        )

    @abstractmethod
    def _build_bounds(self, stats: StatisticService, label: str) -> tuple[Tensor, Tensor]:
        ...