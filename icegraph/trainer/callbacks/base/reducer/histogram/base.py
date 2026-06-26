# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from functools import cached_property
from abc import abstractmethod

import torch
from torch import Tensor

from icegraph.common.tensors import DualResidentTensor
from icegraph.common.histogram import Histogram
from icegraph.common.transforms import TransformSpace
from icegraph.trainer.callbacks.base.accumulator import HistogramAccumulator

from ..reducer import Reducer

__all__ = ["HistogramReducer"]


class HistogramReducer(Reducer[Histogram, HistogramAccumulator]):
    """
    Base class for online data reduction during testing/validation splits.

    Reducers accumulate batch-level data and emit reduced artifacts
    (e.g. histograms) that are later consumed by renderers.
    """

    @cached_property
    def bins(self) -> DualResidentTensor:
        return DualResidentTensor(torch.as_tensor(self._build_bins(), dtype=torch.int64))

    @cached_property
    def scale(self) -> tuple[TransformSpace, ...]:
        scale = self._kwargs.get("scale")

        if scale is not None:
            normalized_scale = tuple(TransformSpace(s) for s in scale)

            # ensure dims match, use bin count for correct dims
            if len(normalized_scale) != self.bins.on("cpu").size(0):
                raise ValueError(
                    f"Invalid scale specification: expected {self.bins.on('cpu').size(0)} entries, "
                    f"got {len(normalized_scale)} ({scale!r})."
                )

            return normalized_scale

        return tuple(TransformSpace.LINEAR for _ in self.bins.on("cpu"))

    @cached_property
    def _expected_ndim(self) -> int:
        return self.bins.on("cpu").size(0)

    def ensure_shape(self, data: torch.Tensor) -> None:
        if data.ndim != 2:
            raise ValueError(
                f"Expected data with shape [B, {self._expected_ndim}], got tensor with shape {tuple(data.shape)}."
            )

        if data.size(1) != self._expected_ndim:
            raise ValueError(
                f"Expected data.shape[1] == {self._expected_ndim}, got {data.size(1)} for shape {tuple(data.shape)}."
            )

    def _flatten(self, t: Tensor, /) -> Tensor:
        # need bins on cpu
        bins = self.bins.on(t.device)

        # compute row-major strides for flattening d-dimensional indices
        strides = torch.ones_like(bins)
        if bins.numel() > 1:
            strides[1:] = torch.cumprod(bins[:-1], dim=0)

        return (t * strides.unsqueeze(0)).sum(dim=-1)

    @cached_property
    def _dense_shape(self) -> tuple[int, ...]:
        return tuple(int(x) for x in self.bins.on("cpu").tolist())

    @cached_property
    def _dense_size(self) -> int:
        return int(torch.prod(self.bins.on("cpu")).item())

    def _to_dense(self, t: Tensor, /) -> Tensor:
        # dense histogram of shape [*bins]
        # minlength forces dense, view unflattens
        return torch.bincount(t, minlength=self._dense_size).view(self._dense_shape)

    def _build_accumulator(self) -> HistogramAccumulator:
        return HistogramAccumulator()

    @abstractmethod
    def _build_bins(self) -> Tensor:
        ...
