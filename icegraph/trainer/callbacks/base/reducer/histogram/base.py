# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

import dataclasses
from abc import abstractmethod
from collections.abc import Mapping
from typing import ClassVar
import math

import torch
from torch import Tensor

from icegraph.common.data import DataRole
from icegraph.common.tensors import DualResidentTensor
from icegraph.common.histogram import Histogram
from icegraph.common.transforms import Space
from icegraph.common.mapping import MemoMap

from ..reducer import Reducer
from .types import HistogramState
from .layout import AxisLayout

__all__ = ["HistogramReducer"]


class HistogramReducer(Reducer[HistogramState, Histogram]):
    """
    Base class for online data reduction during testing/validation splits.

    Reducers accumulate batch-level data and emit reduced artifacts
    (e.g. histograms) that are later consumed by renderers.
    """
    ndim:                       ClassVar[int]   = 1
    has_trailing_count_axis:    ClassVar[bool]  = True

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # memo maps
        self.axis_layout:   MemoMap[str, AxisLayout]                    = MemoMap(self._get_axis_layout)
        self.bins:          MemoMap[str, MemoMap[torch.device, Tensor]] = MemoMap(self._get_bins)
        self.strides:       MemoMap[str, MemoMap[torch.device, Tensor]] = MemoMap(self._get_strides)
        self.size:          MemoMap[str, int]                           = MemoMap(self._get_size)
        self.margins:       MemoMap[str, Tensor]                        = MemoMap(self._get_margins)

    ### Binning ###

    def default_axis_layout(self) -> AxisLayout:
        return AxisLayout()

    def bin_count_override(self) -> tuple[int, ...] | None:
        return None

    def _get_axis_layout(self, label: str) -> AxisLayout:
        # load head index for label
        decoder = self._ctx.engine.services.require("decode", required_by=type(self))
        head_index = decoder.get_columns(DataRole.TARGETS).index(label)

        # load layouts form kwargs
        layouts = self._kwargs.get("layouts", {})

        # attempt to find head-specific layout
        head_layout = layouts.get(label)
        if head_layout is None:
            # return default if also cant find by index
            head_layout = layouts.get(head_index, self.default_axis_layout())

        # if wrong type, break out
        if not isinstance(head_layout, AxisLayout):
            raise TypeError(
                f"{type(self).__name__}: Expected {AxisLayout.__name__} "
                f"for head {head_index} ({label}), got {type(head_layout).__name__}"
            )

        # use bin count override if passed
        bco = self.bin_count_override()
        if bco is not None:
            head_layout = dataclasses.replace(head_layout, bins=bco)

        # expand and return
        return head_layout.expand(type(self).ndim)

    def _get_margins(self, label: str) -> Tensor:
        return torch.as_tensor(self.axis_layout[label].margins, dtype=torch.float32)

    def _get_size(self, label: str) -> int:
        return int(math.prod(self.axis_layout[label].bins))

    def _get_bins(self, label: str) -> MemoMap[torch.device, Tensor]:
        bins = torch.as_tensor(self.axis_layout[label].bins, dtype=torch.int64)
        return MemoMap(lambda d: bins.to(d))

    def _get_strides(self, label: str) -> MemoMap[torch.device, Tensor]:
        bins = self.bins[label][torch.device("cpu")]

        # build strides on cpu
        # row-major, the last axis varies fastest: flat = ... + i0 * (b1 * b2) + i1 * b2 + i2
        strides = torch.ones_like(bins)
        strides[:-1] = torch.cumprod(bins.flip(0), dim=0).flip(0)[1:]

        return MemoMap(lambda d: strides.to(d))

    def get_count_label(self, label: str) -> str:
        if self.axis_layout[label].log_count:
            return r"log<sub>10</sub>(Count)"
        return r"Count"

    def count(self, indices: Tensor, /, label: str) -> Tensor:
        """Dense int64 histogram of shape ``[*bins]`` from per-axis bin indices ``[B, D]``."""
        flat = (indices * self.strides[label][indices.device]).sum(dim=-1)

        assert indices.shape[-1] == self.strides[label][torch.device("cpu")].numel()

        # minlength forces a dense result; strides match the row-major view, so
        # axis order follows the rows (and bounds): 2D data is stored as [x, y]
        return torch.bincount(flat, minlength=self.size[label]).view(self.axis_layout[label].bins)

    ### Monoid ###

    def initial(self) -> HistogramState:
        return None

    def update_state(self, state: HistogramState, rows: Tensor, label: str) -> HistogramState:
        if rows.size(1) != type(self).ndim:
            raise ValueError(f"Expected rows with shape [B, {self.ndim}], got {tuple(rows.shape)}.")
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

        # run subclass reduction
        return {name: self.histogram(t, label) for name, t in self.reduce(counts, label).items()}

    ### Hooks ###

    def reduce(self, counts: Mapping[int, Tensor], label: str) -> Mapping[str, Tensor]:
        """Map per-group counts to named series. Defaults to one series per group."""
        return {self.group_name(key, label): t for key, t in counts.items()}

    def group_name(self, key: int, label: str) -> str:
        """Display name for a group, from the ``class_name_map`` kwarg when given."""
        return self._kwargs.get("class_name_map", {}).get(label, {}).get(key, f"Class {key}")

    @abstractmethod
    def histogram(self, t: Tensor, label: str) -> Histogram:
        ...

    @abstractmethod
    def encode(self, rows: Tensor, label: str) -> Tensor:
        """Bin one batch of rows ``[B, D]`` into a dense count tensor of shape ``[*bins]``."""
        ...
