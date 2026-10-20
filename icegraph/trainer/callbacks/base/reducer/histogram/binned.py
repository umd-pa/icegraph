# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Callable

from abc import abstractmethod

import torch
from torch import Tensor

# local package
from icegraph.common.histogram import Histogram
from icegraph.common.transforms import Space
from icegraph.common.mapping import MemoMap

# local subpackage
from .base import HistogramReducer
from .grid import Grid
from .context import BoundsConstructorContext

__all__ = ["BHistogramReducer"]


_TRANSFORMS: dict[Space, Callable[[Tensor], Tensor]] = {
    Space.LOG: torch.log10,
    Space.ASINH: torch.asinh,
    Space.LINEAR: lambda t: t,
}


class BHistogramReducer(HistogramReducer):
    """Histogram reducer over continuous rows, binned uniformly in each axis' transform space."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.grid: MemoMap[str, MemoMap[torch.device, Grid]] = MemoMap(self._get_grid)

    def _get_grid(self, label: str) -> MemoMap[torch.device, Grid]:
        mins, maxs = self.build_bounds(
            BoundsConstructorContext(self._ctx.engine.decode), label
        )

        if any(t.ndim != 1 for t in (mins, maxs)):
            raise ValueError(
                f"{type(self).__name__}: build_bounds must return tensors of shape "
                f"[{type(self).ndim},], got shapes {mins.shape}, {maxs.shape}."
            )

        # transform into bin space, keeping min <= max (e.g. for decreasing transforms)
        mins, maxs = self.transform(mins, label), self.transform(maxs, label)
        mins, maxs = torch.minimum(mins, maxs), torch.maximum(mins, maxs)

        # apply margins
        pad = (maxs - mins) * self.margins[label]
        mins, maxs = mins - pad, maxs + pad

        grid = Grid(
            mins=mins,
            maxs=maxs,
            scale=self.bins[label][torch.device("cpu")] / (maxs - mins).clamp_min(1e-12)
        )

        return MemoMap(lambda d: grid.to(d))

    def transform(self, t: Tensor, /, label: str) -> Tensor:
        transforms = tuple(_TRANSFORMS[space] for space in self.axis_layout[label].spaces)

        # all the same transform
        if all(f is transforms[0] for f in transforms):
            return transforms[0](t)

        if t.shape[-1] != len(transforms):
            raise ValueError(f"Expected last dimension size {len(transforms)}, got {t.shape[-1]}.")

        return torch.stack(
            [f(part) for part, f in zip(t.unbind(dim=-1), transforms, strict=True)],
            dim=-1,
        )

    def encode(self, rows: Tensor, label: str) -> Tensor:
        rows = self.transform(rows, label)
        grid = self.grid[label][rows.device]
        bins = self.bins[label][rows.device]

        # keep rows within bounds, including the upper edge
        rows = rows[((rows >= grid.mins) & (rows <= grid.maxs)).all(dim=-1)]

        # continuous -> discrete bin indices
        indices = torch.floor((rows - grid.mins) * grid.scale).to(torch.int64)
        indices = indices.clamp(min=0).minimum(bins - 1)

        return self.count(indices, label)

    def histogram(self, t: Tensor, label: str) -> Histogram:
        grid = self.grid[label][torch.device("cpu")]
        return Histogram(histogram=t.cpu().numpy(), bounds=torch.stack((grid.mins, grid.maxs)).numpy())

    @abstractmethod
    def build_bounds(self, ctx: BoundsConstructorContext, label: str) -> tuple[Tensor, Tensor]:
        ...
