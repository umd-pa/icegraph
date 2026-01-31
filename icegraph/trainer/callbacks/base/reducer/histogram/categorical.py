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
from icegraph.trainer.callbacks.base.accumulator import Accumulator, AccumulatorStore, dense_histogram_accumulator

# local subpackage
from .histogram import HistogramReducer
from ._utils import build_label_index_map, flatten

if TYPE_CHECKING:
    from icegraph.trainer import Trainer

__all__ = ["CategoricalHistogramReducer"]


class CategoricalHistogramReducer(HistogramReducer, ABC):
    # accumulator names
    CORE = "core"

    # accumulator dtypes
    CORE_DTYPE = torch.long

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # params required
        self._bins: DualResidentTensor | None = None

    def _post_init(self, trainer: Trainer) -> None:
        self._bins = DualResidentTensor(self._build_bins())

        # assertions
        assert self._bins is not None

    @abstractmethod
    def _build_bins(self) -> Tensor:
        ...

    def _build_accumulators(self, device: torch.device) -> dict[str, Accumulator]:
        # grab params on correct device
        bx, by = self._bins.on("cpu").tolist()

        # the core accumulator is a 3D dense histogram with shape (label_count, nbin_y, nbin_x)
        accumulators: dict[str, Accumulator] = {
            self.CORE: dense_histogram_accumulator((len(self._target_labels), by, bx), device, self.CORE_DTYPE)
        }

        return accumulators
    
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

    def _build_payload(self, indices: Tensor, masks: dict[str, Tensor], index_map: Tensor) -> dict[str, Tensor]:
        payload: dict[str, Tensor] = {
            self.CORE: self._encode_c(indices, masks[self.CORE], index_map)
        }

        return payload

    def _encode(self, indices: Tensor) -> dict[str, Tensor]:
        # build masks
        masks = self._build_masks(indices)

        # build label index mapping
        label_index_map = build_label_index_map(indices)

        payload = self._build_payload(indices, masks, label_index_map)

        return payload

    def _build_masks(self, indices: Tensor) -> dict[str, Tensor]:
        # grab device from indices
        device = indices.device

        # grab bins on device
        bins_c_cuda = self._bins.on(device)

        # build mask for core
        masks: dict[str, Tensor] = {self.CORE: ((indices >= 0) & (indices < bins_c_cuda)).all(dim=-1)}

        # fail fast if any indices are out of bounds
        # categorical histograms cannot fundamentally have out of bound indices, if they do that is a big problem
        if not all(mask.all().item() for mask in masks.values()):
            raise ValueError(
                f"Out-of-bound indices detected in {type(self).__name__}. "
                f"This likely indicates invalid class predictions from the model "
                f"or a fault in runtime data processing."
            )

        return masks

    def _emit_artifacts(self, trainer: Trainer, accumulator: AccumulatorStore) -> Iterator[Histogram]:
        # enumerate all accumulators and build/yield histogram objects
        for index, (core,) in accumulator.enum_all():
            # core must exist for each label
            assert core is not None, f"Missing core histogram for label {self._target_labels[index]}."

            histogram = Histogram(
                name=self._target_labels[index],
                histogram=core.cpu().numpy(),
                space=self._axis_scale
            )
            yield histogram