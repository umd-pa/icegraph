# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from functools import cached_property, lru_cache
from typing import Any, ClassVar

import torch
from torch import Tensor
import numpy as np
from jaxtyping import Int, Float

from icegraph.statistics import StatisticService
from icegraph.common.data import DataRole, Split, ColumnarRole
from icegraph.common.record import Record
from icegraph.typing.common import ArrayI
from icegraph.common.tensors import SegmentLayout

from ..service import Service

from .config import DecodeConfig
from .attrs import AttributeDecoder, AttributeDecoderFactory, AttributeDecoderContext
from .records import RecordDecoder, RecordDecoderFactory, RecordDecoderContext

import logging
logger = logging.getLogger(__name__)

__all__ = ["DecodeService"]


class DecodeService(Service[DecodeConfig]):
    name: ClassVar[str] = "decode"
    version: ClassVar[int] = 1

    def build(self) -> None:
        return

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> DecodeConfig:
        return DecodeConfig(**config)

    ### ENUM TO KEY

    def _role_to_key(self, role: ColumnarRole) -> str:
        mapping = {
            DataRole.FEATURES:      self.config.keymap.features,
            DataRole.TARGETS:       self.config.keymap.truth,
            DataRole.AUXILIARY:     self.config.keymap.truth
        }

        return mapping[role]
    
    def _split_to_key(self, split: Split) -> int:
        return split.to_int()

    ### INDEXING

    @lru_cache(maxsize=None)
    def _logical_indices(self, role: ColumnarRole) -> ArrayI:
        normalized_key = self._role_to_key(role)

        available = self._attr_decoder.extract_columns(normalized_key)  # raw, not cached
        requested = {
            DataRole.FEATURES: self.config.features,
            DataRole.TARGETS: self.config.targets,
            DataRole.AUXILIARY: self.config.auxiliary
        }[role]

        # if this is for targets or features, and user passed nothing, assume all columns
        if role in DataRole.core() and not requested:
            return np.arange(len(available)).astype(np.int64, copy=False)

        pos = {name: i for i, name in enumerate(available)}
        missing = [c for c in requested if c not in pos]
        if missing:
            raise ValueError(
                f"{type(self).__name__}: unknown column(s) for role {role.name}: "
                f"{missing}; available: {available}"
            )

        if len(set(requested)) != len(requested):
            raise ValueError(
                f"{type(self).__name__}: duplicate column(s) for role {role.name}: {requested}"
            )

        # order is not meaningful; canonicalize to sorted (file order) for global consistency
        indices = np.fromiter((pos[c] for c in requested), dtype=np.int64)
        indices.sort()
        return indices.astype(np.int64, copy=False)  # no-op, but helps type correctly

    @lru_cache(maxsize=None)
    def _indices(self, role: ColumnarRole) -> Tensor:
        normalized_key = self._role_to_key(role)

        logical = self._logical_indices(role)
        offsets = self._attr_decoder.extract_offsets(normalized_key)  # raw group boundaries

        # expand each selected logical group into its physical column range
        physical = np.concatenate([
            np.arange(offsets[i], offsets[i + 1], dtype=np.int64)
            for i in logical
        ]) if len(logical) else np.empty(0, dtype=np.int64)

        return torch.tensor(physical)

    ### DECODERS

    @cached_property
    def _attr_decoder(self) -> AttributeDecoder[Any]:
        record = self._ctx.services.require("record", required_by=type(self))

        # build the decoder
        decoder = AttributeDecoderFactory.create(self.config.attrs.name, **self.config.attrs.kwargs)

        # attach the decoder
        ctx = AttributeDecoderContext(attrs=record.attrs, global_attrs=record.global_attrs)
        decoder.attach(ctx)

        return decoder

    @cached_property
    def _record_decoder(self) -> RecordDecoder[Any]:
        # build the decoder
        decoder = RecordDecoderFactory.create(self.config.records.name, **self.config.records.kwargs)

        # attach the decoder
        ctx = RecordDecoderContext()
        decoder.attach(ctx)

        return decoder

    ### ATTRIBUTE DECODER HOOKS

    @lru_cache(maxsize=None)
    def get_columns(self, role: ColumnarRole) -> list[str]:
        normalized_key = self._role_to_key(role)

        raw = self._attr_decoder.extract_columns(normalized_key)
        indices: list[int] = self._logical_indices(role).tolist()

        return [raw[i] for i in indices]

    @lru_cache(maxsize=None)
    def get_offsets(self, role: ColumnarRole) -> Tensor:
        normalized_key = self._role_to_key(role)

        raw = torch.as_tensor(self._attr_decoder.extract_offsets(normalized_key))  # cumulative, full physical layout
        logical = torch.as_tensor(self._logical_indices(role).copy(), dtype=torch.long)  # sorted group selection

        # widths of selected groups
        widths = torch.diff(raw)[logical]

        # empty 0 filled tensor
        filtered = torch.zeros(len(logical) + 1, dtype=raw.dtype)

        # recompute boundaries from 0
        filtered[1:] = torch.cumsum(widths, dim=0)
        return filtered

    @lru_cache(maxsize=None)
    def get_stats(self, split: Split, role: ColumnarRole) -> StatisticService:
        normalized_key = (self._split_to_key(split), self._role_to_key(role))

        stats = self._attr_decoder.extract_stats(normalized_key)   # raw, per physical column

        # convert physical index selection to boolean mask over the full physical width
        n_physical = stats.num_columns()
        mask = np.zeros(n_physical, dtype=bool)
        mask[self._indices(role).numpy()] = True

        # filter the stats to match selection
        stats.filter_to(mask)
        return stats

    @lru_cache(maxsize=None)
    def get_keys(self, split: Split) -> ArrayI:
        normalized_key = self._split_to_key(split)
        return self._attr_decoder.extract_keys(normalized_key)  # not column filtered

    @lru_cache(maxsize=None)
    def get_segment_layout(self, role: ColumnarRole, device: torch.device) -> SegmentLayout:
        # build segment layout
        return SegmentLayout.build(
            offsets=self.get_offsets(role),
            names=self.get_columns(role)
        ).to(device)

    @cached_property
    def count_by_weight_group(self) -> dict[str, int]:
        return self._attr_decoder.extract_count_by_weight_group()

    ### RECORD DECODER HOOKS
    # the excluded parameter, despite seemingly redundant, allows this service to fully control
    # the structure of empty roles

    def _empty_tensor(self, shape: tuple[int, ...], dtype: torch.dtype) -> Tensor:
        return torch.empty(shape, dtype=dtype)

    def load_features(self, record: Record, excluded: bool = False) -> Float[Tensor, "N F"]:
        if excluded:
            return self._empty_tensor((0,), dtype=torch.float32)

        key = self.config.keymap.features
        raw = self._record_decoder.extract_features(record, key)      # [N, F_pre]

        if raw is None:
            return self._empty_tensor((0,), dtype=torch.float32)

        return raw[..., self._indices(DataRole.FEATURES)]          # [N, F]

    def load_targets(self, record: Record, excluded: bool = False) -> Float[Tensor, "1 T"] | Int[Tensor, "1 T"]:
        if excluded:
            return self._empty_tensor((1, 0), dtype=torch.float32)

        key = self.config.keymap.truth
        raw = self._record_decoder.extract_targets(record, key)         # [1, T_pre]

        if raw is None:
            return self._empty_tensor((1, 0), dtype=torch.float32)

        return raw[..., self._indices(DataRole.TARGETS)]           # [1, T]

    def load_auxiliary(self, record: Record, excluded: bool = False) -> Float[Tensor, "1 A"] | Int[Tensor, "1 A"]:
        if excluded:
            return torch.empty((1, 0), dtype=torch.float32)

        key = self.config.keymap.truth
        raw = self._record_decoder.extract_auxiliary(record, key)       # [1, A_pre]

        if raw is None:
            return self._empty_tensor((1, 0), dtype=torch.float32)

        return raw[..., self._indices(DataRole.AUXILIARY)]         # [1, A]

    def load_edge_index(self, record: Record, excluded: bool = False) -> Int[Tensor, "2 E"]:
        if excluded:
            return self._empty_tensor((2, 0), dtype=torch.long)

        key = self.config.keymap.edge_index
        raw = self._record_decoder.extract_edge_index(record, key)

        if raw is None:
            return self._empty_tensor((2, 0), dtype=torch.long)

        return raw

    def load_edge_attr(self, record: Record, excluded: bool = False) -> Float[Tensor, "E ATTR"]:
        if excluded:
            return self._empty_tensor((0,), dtype=torch.float32)

        key = self.config.keymap.edge_attr
        raw = self._record_decoder.extract_edge_attr(record, key)

        if raw is None:
            return self._empty_tensor((0,), dtype=torch.float32)

        return raw

    def load_simweights(self, record: Record, excluded: bool = False) -> Float[Tensor, "1"] | Float[Tensor, "0"]:
        if excluded:
            return self._empty_tensor((0,), dtype=torch.float32)

        key = self.config.keymap.simweights
        raw = self._record_decoder.extract_simweights(record, key)

        if raw is None:
            return self._empty_tensor((0,), dtype=torch.float32)

        return raw
