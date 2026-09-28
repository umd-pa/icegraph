# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from functools import cached_property
from abc import abstractmethod
from collections.abc import Mapping

import torch
from torch import Tensor

from icegraph.common.tensors import DualResidentTensor
from icegraph.common.histogram import Histogram
from icegraph.common.transforms import TransformSpace

from ..reducer import Reducer
from .types import HistogramState

__all__ = ["HistogramReducer"]


class HistogramReducer(Reducer[HistogramState, Histogram]):
    """
    Base class for online data reduction during testing/validation splits.

    Reducers accumulate batch-level data and emit reduced artifacts
    (e.g. histograms) that are later consumed by renderers.
    """

    ### Binning ###

    @cached_property
    def bins(self) -> DualResidentTensor:
        return DualResidentTensor(torch.as_tensor(self._build_bins(), dtype=torch.int64))

    @cached_property
    def ndim(self) -> int:
        return self.bins.on("cpu").numel()

    @cached_property
    def scale(self) -> tuple[TransformSpace, ...]:
        scale = self._kwargs.get("scale")

        if scale is None:
            return (TransformSpace.LINEAR,) * self.ndim

        normalized = tuple(TransformSpace(s) for s in scale)

        if len(normalized) != self.ndim:
            raise ValueError(
                f"Invalid scale specification: expected {self.ndim} entries, got {len(normalized)} ({scale!r})."
            )

        return normalized

    @cached_property
    def _strides(self) -> DualResidentTensor:
        # row-major, the last axis varies fastest: flat = ... + i0 * (b1 * b2) + i1 * b2 + i2
        bins = self.bins.on("cpu")
        strides = torch.ones_like(bins)
        strides[:-1] = torch.cumprod(bins.flip(0), dim=0).flip(0)[1:]
        return DualResidentTensor(strides)

    @cached_property
    def _dense_shape(self) -> tuple[int, ...]:
        return tuple(self.bins.on("cpu").tolist())

    @cached_property
    def _dense_size(self) -> int:
        return int(self.bins.on("cpu").prod().item())

    def _check_rows(self, rows: Tensor, /) -> None:
        if rows.size(1) != self.ndim:
            raise ValueError(f"Expected rows with shape [B, {self.ndim}], got {tuple(rows.shape)}.")

    def _count(self, indices: Tensor, /) -> Tensor:
        """Dense int64 histogram of shape ``[*bins]`` from per-axis bin indices ``[B, D]``."""
        flat = (indices * self._strides.on(indices.device)).sum(dim=-1)

        # minlength forces a dense result; strides match the row-major view, so
        # axis order follows the rows (and bounds): 2D data is stored as [x, y]
        return torch.bincount(flat, minlength=self._dense_size).view(self._dense_shape)

    ### Monoid ###

    def initial(self) -> HistogramState:
        return None

    def update_state(self, state: HistogramState, rows: Tensor, label: str) -> HistogramState:
        self._check_rows(rows)
        return self.combine(state, self.encode(rows, label))

    def combine(self, a: HistogramState, b: HistogramState) -> HistogramState:
        # trivial cases
        if a is None:
            return b
        if b is None:
            return a

        return a + b.to(a.device)

    def finalize(self, states: Mapping[int, HistogramState], label: str) -> dict[str, Histogram]:
        counts = {k: s for k, s in states.items() if s is not None}
        return {name: self._histogram(t, label) for name, t in self.reduce(counts, label).items()}

    ### Hooks ###

    def reduce(self, counts: Mapping[int, Tensor], label: str) -> Mapping[str, Tensor]:
        """Map per-group counts to named series. Defaults to one series per group."""
        return {self.group_name(key, label): t for key, t in counts.items()}

    def group_name(self, key: int, label: str) -> str:
        """Display name for a group, from the ``class_name_map`` kwarg when given."""
        return self._kwargs.get("class_name_map", {}).get(label, {}).get(key, f"Class {key}")

    def _histogram(self, t: Tensor, label: str) -> Histogram:
        return Histogram(histogram=t.cpu().numpy())

    @abstractmethod
    def encode(self, rows: Tensor, label: str) -> Tensor:
        """Bin one batch of rows ``[B, D]`` into a dense count tensor of shape ``[*bins]``."""
        ...

    @abstractmethod
    def _build_bins(self) -> Tensor:
        ...
