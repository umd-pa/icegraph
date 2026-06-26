# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Callable, Any
from abc import abstractmethod, ABC

import torch
from torch import Tensor

# local package
from icegraph.statistics import StatisticService
from icegraph.common.transforms import TransformSpace
from icegraph.common.data import DataRole, Split, ColumnarRole
from icegraph.common.engine import ComponentKind
from icegraph.common.tensors import SegmentedTensor

from icegraph.engine.components.normalizer import Normalizer

from .config import Config

__all__ = ["AffineNormalizer"]

# module logger
import logging
logger = logging.getLogger(__name__)


class AffineNormalizer(Normalizer[Config], ABC):

    def build(self) -> None:
        """Initialize the normalizer."""
        # for each of scale/offset and for each data role, create empty buffer
        for role in DataRole.columnar():
            self.register_dynamic_buffer(f"_scale_{role.name}", None)
            self.register_dynamic_buffer(f"_offset_{role.name}", None)

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> Config:
        return Config(**config)

    def scale(self, role: ColumnarRole) -> Tensor:
        buffer_name = f"_scale_{role.name}"

        # load buffer
        scale = self.load_buffer(buffer_name, allow_none=True)

        # if not built, build
        if scale is None:
            scale = self._resolve(role, self._build_scale)

            # register the buffer
            self.register_buffer(buffer_name, scale)

        return scale

    def offset(self, role: ColumnarRole) -> Tensor:
        buffer_name = f"_offset_{role.name}"

        # load buffer
        offset = self.load_buffer(buffer_name, allow_none=True)

        # if not built, build
        if offset is None:
            offset = self._resolve(role, self._build_offset)

            # register the buffer
            self.register_buffer(buffer_name, offset)

        return offset

    def transformer_spec_list(self, role: ColumnarRole):  # type hinter can derive from transformer.spec_list so ignore
        transformer = self._ctx.components.require(ComponentKind.TRANSFORMER, required_by=type(self))
        return transformer.spec_list(role)

    def _resolve(
            self,
            role: ColumnarRole,
            build: Callable[[StatisticService, TransformSpace, int, int], float],
    ) -> Tensor:
        # decoder service
        decoder = self._ctx.services.require("decode", required_by=type(self))

        # get stats, only want training stats, we don't care about val or test
        stats = decoder.get_stats(Split.TRAIN, role)

        # build param list
        params = []
        for column_index, spec in enumerate(self.transformer_spec_list(role)):
            # get value from build method
            value = build(stats, spec.space, spec.base, column_index)

            # ensure of type float
            if not isinstance(value, float):
                raise TypeError(
                    f"Expected build(...) to return a float for column_index={column_index}, "
                    f"but got {type(value).__name__}: {value!r}"
                )

            # append to params
            params.append(value)

        # convert to tensor and return
        return torch.tensor(params, dtype=torch.float32, device=self.device)

    @abstractmethod
    def _build_scale(self, stats: StatisticService, space: TransformSpace, base: int, column_index: int) -> float:
        ...

    @abstractmethod
    def _build_offset(self, stats: StatisticService, space: TransformSpace, base: int, column_index: int) -> float:
        ...

    def normalize(self, t: SegmentedTensor, /, role: ColumnarRole, *, inverse: bool = False) -> Tensor:
        """Forward pass through the normalizer."""
        scale = self.scale(role)
        offset = self.offset(role)

        if inverse:
            return t.data.div(scale).add(offset)
        return t.data.add(-offset).mul(scale)
