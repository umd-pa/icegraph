# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Callable

from abc import abstractmethod
from functools import cached_property

import torch
from torch import Tensor

# local package
from icegraph.common.histogram import Histogram
from icegraph.common.tensors import DualResidentTensor
from icegraph.common.data import Split, DataRole
from icegraph.statistics import StatisticService
from icegraph.common.transforms import TransformSpace

# local subpackage
from .base import HistogramReducer
from .types import Binning

__all__ = ["BHistogramReducer"]


_TRANSFORMS: dict[TransformSpace, Callable[[Tensor], Tensor]] = {
    TransformSpace.LOG: torch.log10,
    TransformSpace.ASINH: torch.asinh,
    TransformSpace.LINEAR: lambda t: t,
}


class BHistogramReducer(HistogramReducer):
    """Histogram reducer over continuous rows, binned uniformly in each axis' transform space."""

    @cached_property
    def transforms(self) -> tuple[Callable[[Tensor], Tensor], ...]:
        return tuple(_TRANSFORMS[space] for space in self.scale)

    def transform(self, t: Tensor, /) -> Tensor:
        transforms = self.transforms

        if all(f is transforms[0] for f in transforms):
            return transforms[0](t)

        if t.shape[-1] != len(transforms):
            raise ValueError(f"Expected last dimension size {len(transforms)}, got {t.shape[-1]}.")

        return torch.stack(
            [f(part) for part, f in zip(t.unbind(dim=-1), transforms, strict=True)],
            dim=-1,
        )

    @cached_property
    def _binnings(self) -> dict[str, Binning]:
        return {}

    def binning(self, label: str) -> Binning:
        """Bounds and bin scale for ``label`` in transformed space, built on first use."""
        binning = self._binnings.get(label)

        if binning is None:
            binning = self._binnings[label] = self._build_binning(label)

        return binning

    def _build_binning(self, label: str) -> Binning:
        # pass a copy of the train stats downstream
        stats = self._ctx.engine.decode.get_stats(Split.TRAIN, DataRole.TARGETS).copy()

        mins, maxs = self._build_bounds(stats, label)

        # transform into bin space, keeping min <= max (e.g. for decreasing transforms)
        mins, maxs = self.transform(mins), self.transform(maxs)
        mins, maxs = torch.minimum(mins, maxs), torch.maximum(mins, maxs)

        mins, maxs = self._apply_margin(mins, maxs)

        return Binning(
            mins=DualResidentTensor(mins),
            maxs=DualResidentTensor(maxs),
            scale=DualResidentTensor(self.bins.on("cpu") / (maxs - mins).clamp_min(1e-12)),
        )

    def _apply_margin(self, mins: Tensor, maxs: Tensor) -> tuple[Tensor, Tensor]:
        margin = self._kwargs.get("margin")
        if margin is None:
            return mins, maxs

        span = maxs - mins  # [d]

        margin = torch.as_tensor(margin, dtype=span.dtype, device=span.device)

        if margin.numel() != 2 * span.numel():
            raise ValueError(f"Expected {2 * span.numel()} margin values, got {margin.numel()}.")

        lower, upper = margin.view(span.numel(), 2).T  # [d] each

        return mins - span * lower, maxs + span * upper

    def encode(self, rows: Tensor, label: str) -> Tensor:
        rows = self.transform(rows)

        mins, maxs, scale = self.binning(label).on(rows.device)

        # keep rows within bounds, including the upper edge
        rows = rows[((rows >= mins) & (rows <= maxs)).all(dim=-1)]

        # continuous -> discrete bin indices
        indices = torch.floor((rows - mins) * scale).to(torch.int64)
        indices = indices.clamp(min=0).minimum(self.bins.on(rows.device) - 1)

        return self._count(indices)

    def _histogram(self, t: Tensor, label: str) -> Histogram:
        mins, maxs, _ = self.binning(label).on("cpu")
        return Histogram(histogram=t.cpu().numpy(), bounds=torch.stack((mins, maxs)).numpy())

    @abstractmethod
    def _build_bounds(self, stats: StatisticService, label: str) -> tuple[Tensor, Tensor]:
        ...
