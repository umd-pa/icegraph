# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Iterator

import torch
from torch import Tensor

# local package
from icegraph.common.histogram import Histogram
from icegraph.common.tensors import DualResidentTensor
from icegraph.trainer.callbacks.base.accumulator import (
    Accumulator, AccumulatorStore, dense_histogram_accumulator, sparse_histogram_accumulator
)
from icegraph.statistics import StatisticService

# local subpackage
from .histogram import HistogramReducer
from ._utils import build_label_index_map, flatten

if TYPE_CHECKING:
    from icegraph.trainer import Trainer

import logging
logger = logging.getLogger(__name__)


class BinnedHistogramReducer(HistogramReducer, ABC):
    # accumulator names
    CORE        = "core"
    EXTENDED    = "extended"
    OVERFLOW    = "overflow"

    # accumulator dtypes
    CORE_DTYPE      = torch.long
    EXTENDED_DTYPE  = torch.long
    OVERFLOW_DTYPE  = torch.long

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # params required
        self._bins:     DualResidentTensor | None = None
        self._margin:   DualResidentTensor | None = None
        self._bounds:   DualResidentTensor | None = None
        self._scale:    DualResidentTensor | None = None

    def _post_init(self, trainer: Trainer) -> None:
        self._bins      = DualResidentTensor(self._build_bins())
        self._margin    = DualResidentTensor(self._build_margin())

        # bounds needs to be transformed appropriately
        _bounds = self._build_bounds(self._stats.copy())

        # temporarily move to device for transform, then return after
        _bounds = _bounds.to(self.device)
        _bounds = self._transform(_bounds)
        _bounds = _bounds.to("cpu")

        self._bounds    = DualResidentTensor(_bounds)
        self._scale     = DualResidentTensor(self._build_scale())

        # assertions
        assert self._bins is not None
        assert self._margin is not None
        assert self._bounds is not None
        assert self._scale is not None

    @abstractmethod
    def _build_bins(self) -> Tensor:
        ...

    @abstractmethod
    def _build_bounds(self, stats: StatisticService) -> Tensor:
        ...

    @abstractmethod
    def _build_margin(self) -> Tensor:
        ...

    def _build_scale(self) -> Tensor:
        # grab bounds on correct device
        bounds_cpu = self._bounds.on("cpu")

        # scale = nbins / (max - min) for each axis
        numerator = self._bins.on("cpu") - 1
        denominator = bounds_cpu[:, 1, :] - bounds_cpu[:, 0, :]

        # clamp to avoid div by 0
        return numerator / denominator.clamp_min(1e-12)

    def _post_reduce(self, data: Tensor) -> Tensor:
        # obtain device from data
        device = data.device

        # run transformation on data
        self._transform(data)

        # compute continuous bin indices
        mins    = self._bounds.on(device)[:, 0, :].unsqueeze(0)
        scale   = self._scale.on(device).unsqueeze(0)
        indices = (data - mins) * scale

        # floor to get discrete indices
        indices = torch.floor_(indices).to(torch.long)

        return indices

    def _build_accumulators(self, device: torch.device) -> dict[str, Accumulator]:
        # grab params on correct device
        bx, by = self._bins.on("cpu").tolist()

        # the core accumulator is a 3D dense histogram with shape (label_count, nbin_y, nbin_x)
        # extended is sparse histogram with ndim=2
        # overflow is a scalar for each label, so dense histogram with size (label_count,)
        accumulators: dict[str, Accumulator] = {
            self.CORE:      dense_histogram_accumulator((len(self._target_labels), by, bx), device, self.CORE_DTYPE),
            self.EXTENDED:  sparse_histogram_accumulator(2, device, self.EXTENDED_DTYPE),
            self.OVERFLOW:  dense_histogram_accumulator((len(self._target_labels),), device, self.OVERFLOW_DTYPE)
        }

        return accumulators

    def _build_masks(self, indices: Tensor) -> dict[str, Tensor]:
        # grab device from indices
        device = indices.device

        # grab bins on device
        bins_c_cuda = self._bins.on(device)

        # init masks dict
        masks: dict[str, Tensor] = {}

        # build mask for core
        masks[self.CORE] = ((indices >= 0) & (indices < bins_c_cuda)).all(dim=-1)

        # get extended region
        bins_e_cuda = self._margin.on(device) * bins_c_cuda

        # within extended region but not in core
        mask_ec = ((indices >= -bins_e_cuda) & (indices < bins_e_cuda)).all(dim=-1)
        masks[self.EXTENDED] = mask_ec & (~masks[self.CORE])

        # not in extended region or core
        masks[self.OVERFLOW] = ~mask_ec

        return masks
    
    def _encode_c(self, indices: Tensor, mask: Tensor, index_map: Tensor) -> Tensor:
        # grab label count from indices
        label_count = indices.size(1)

        # apply mask and flatten core
        flat_c = flatten(indices[mask], index_map[mask].squeeze(-1), self._bins.on(indices.device))

        # minlength ensures a non-sparse result, as core should be dense
        bins_c_cpu = self._bins.on("cpu")
        return (torch
            .bincount(flat_c, minlength=label_count * torch.prod(bins_c_cpu).item())
            .view(label_count, bins_c_cpu[1].item(), bins_c_cpu[0].item())
            .to(dtype=self.CORE_DTYPE)
        )

    def _encode_e(self, indices: Tensor, mask: Tensor, index_map: Tensor) -> Tensor:
        # for extended region, do things a little different
        tuples_e = torch.cat([index_map, indices], dim=-1)[mask].reshape(-1, 3)

        # count unique values using torch.unique, not using bincount here
        unique_e, count_e = torch.unique(tuples_e, dim=0, return_counts=True)

        return (torch
            .cat([unique_e, count_e.unsqueeze(1)], dim=1)
            .to(dtype=self.EXTENDED_DTYPE)
        )

    def _encode_o(self, indices: Tensor, mask: Tensor, index_map: Tensor) -> Tensor:
        label_count = indices.size(1)

        # build flattened 1D tensor with an entry for each overflow under each label
        overflow_labels = index_map[mask].reshape(-1)

        # minlength ensures a dense result in case one or more labels have no overflow
        return (torch
            .bincount(overflow_labels, minlength=label_count)
            .to(dtype=self.OVERFLOW_DTYPE)
        )

    def _build_payload(self, indices: Tensor, masks: dict[str, Tensor], index_map: Tensor) -> dict[str, Tensor]:
        payload: dict[str, Tensor] = {
            self.CORE:      self._encode_c(indices, masks[self.CORE], index_map),
            self.EXTENDED:  self._encode_e(indices, masks[self.EXTENDED], index_map),
            self.OVERFLOW:  self._encode_o(indices, masks[self.OVERFLOW], index_map)
        }

        return payload

    def _encode(self, indices: Tensor) -> dict[str, Tensor]:
        # build masks
        masks = self._build_masks(indices)

        # build label index mapping
        label_index_map = build_label_index_map(indices)

        payload = self._build_payload(indices, masks, label_index_map)

        return payload

    def _emit_artifacts(self, trainer: Trainer, accumulator: AccumulatorStore) -> Iterator[Histogram]:
        # align for proper ordering
        accumulator.align_to([self.CORE, self.EXTENDED, self.OVERFLOW])

        # get bounds on cpu
        bounds_cpu = self._bounds.on("cpu")

        # enumerate all accumulators and build/yield histogram objects
        for index, (core, extended, overflow) in accumulator.enum_all():
            # core must exist for each label
            assert core is not None, f"Missing core histogram for label {self._target_labels[index]}."

            histogram = Histogram(
                name=self._target_labels[index],
                histogram=core.cpu().numpy(),
                space=self._axis_scale,
                bounds=bounds_cpu[index].transpose(0, 1).numpy(),
                extended=extended.cpu().numpy() if (extended is not None and extended.numel() > 0) else None,
                overflow=overflow.cpu().numpy() if (overflow is not None and overflow.numel() > 0) else None
            )
            yield histogram