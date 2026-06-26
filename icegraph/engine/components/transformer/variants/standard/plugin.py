# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Any, ClassVar
from functools import lru_cache

import torch
from torch import Tensor

from icegraph.common.transforms import TransformSpace
from icegraph.common.data import DataRole, ColumnarRole
from icegraph.common.tensors import SegmentedTensor
from icegraph.engine.components.transformer import Transformer
from icegraph.engine.components.transformer.types import TransformerSpec

from .factory import TransformerModuleFactory
from .module import TransformerModule
from .config import TransformerConfig

__all__ = ["StandardTransformer"]

import logging
logger = logging.getLogger(__name__)


class StandardTransformer(Transformer[TransformerConfig]):
    name: ClassVar[str] = "standard"
    version: ClassVar[int] = 1

    def build(self) -> None:
        # for each non-linear space, create empty buffer for mapping and log_base
        for role in DataRole.columnar():
            # each role needs a log_base buffer
            self.register_dynamic_buffer(f"_log_base_{role.name}", None)

            # each role needs a mapping for each nonlinear space
            for space in TransformSpace.non_linear():
                self.register_dynamic_buffer(f"_mapping_{role.name}_{space.name}", None)

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> TransformerConfig:
        return TransformerConfig(**config)

    def _build_spec_list(self, role: ColumnarRole) -> list[TransformerSpec]:
        # data service
        decoder = self._ctx.services.require("decode", required_by=type(self))

        # load directly from decoder for init time construction
        segment_layout = decoder.get_segment_layout(role, self.device)

        # pull offsets and name from segment layout
        names = segment_layout.names
        offsets = segment_layout.offsets

        # build spec list for role
        specs: list[TransformerSpec] = []
        for logic_index, name in enumerate(names):
            # default to linear (base 10 here is meaningless for linear, just keeping it as 10 for convention)
            spec = TransformerSpec(TransformSpace.LINEAR, 10)

            # load column config if available
            if (config := self.config.transforms.get(name)) is not None:
                spec = TransformerSpec(TransformSpace(config.space), config.base)

            # append the spec for each sub column
            for _ in range(offsets[logic_index], offsets[logic_index + 1]):
                # immutable, so can append same instance repeatedly
                specs.append(spec)

        return specs

    @lru_cache(maxsize=None)
    def transformer_module(self, space: TransformSpace) -> TransformerModule:
        return TransformerModuleFactory.create(space.value)

    def log_base(self, role: ColumnarRole) -> Tensor:
        buffer_name = f"_log_base_{role.name}"

        # load buffer
        log_base = self.load_buffer(buffer_name, allow_none=True)

        # if not built, build
        if log_base is None:
            # build the full log-of-base list for the role
            base = torch.tensor(
                [spec.base for spec in self.spec_list(role)],
                dtype=torch.float32,
                device=self.device
            )

            # take natural log of each base
            log_base = torch.log(base)

            # register the buffer
            self.register_buffer(buffer_name, log_base)

        return log_base

    def mapping(self, role: ColumnarRole, space: TransformSpace) -> Tensor:
        buffer_name = f"_mapping_{role.name}_{space.name}"

        # load buffer
        mapping = self.load_buffer(buffer_name, allow_none=True)

        # if not built, build
        if mapping is None:
            # build the col mapping
            mapping = torch.tensor(
                [col for col, spec in enumerate(self.spec_list(role)) if spec.space == space],
                dtype=torch.long,
                device=self.device
            )

            # register the buffer
            self.register_buffer(buffer_name, mapping)

        return mapping

    def transform(self, t: SegmentedTensor, /, role: ColumnarRole, *, inverse: bool) -> Tensor:
        out = t.data.clone()

        # load log_base buffer
        log_base = self.log_base(role)

        for space in TransformSpace.non_linear():
            # load buffer
            mapping = self.mapping(role, space)

            # skip space for empty cols
            if mapping.numel() == 0:
                continue

            # filter log_base to cols
            log_base_selection = log_base.index_select(dim=0, index=mapping)

            # filter out to cols
            selection = out.index_select(dim=-1, index=mapping)

            # run vectorized transform
            module = self.transformer_module(space)
            selection = module.forward(
                selection, log_base=log_base_selection, inverse=inverse
            )

            # copy changes back to out tensor
            out.index_copy_(dim=-1, index=mapping, source=selection)

        return out
