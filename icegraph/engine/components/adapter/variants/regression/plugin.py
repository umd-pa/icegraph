# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import ClassVar, Any

import torch
from torch import Tensor

from icegraph.common.data import DataRole, GraphBatch
from icegraph.engine.components.adapter import Adapter

from .config import RegressionConfig

__all__ = ["Regression"]


class Regression(Adapter[RegressionConfig]):
    name: ClassVar[str] = "regression"
    version: ClassVar[int] = 1

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> RegressionConfig:
        return RegressionConfig(**config)

    def model_forward_validator(self, t: Tensor, /, debug: bool) -> None:
        expected_c = self.get_out_channels()

        # structural checks, metadata only, no GPU sync
        if t.ndim != 2:
            raise ValueError(
                f"{type(self).__name__}: expected model output of rank 2 "
                f"[B, C], got rank {t.ndim} (shape {tuple(t.shape)})."
            )
        if t.shape[-1] != expected_c:
            raise ValueError(
                f"{type(self).__name__}: expected {expected_c} output channels "
                f"(out_channels), got {t.shape[-1]} (shape {tuple(t.shape)})."
            )

        # forces a sync, debug only
        if debug:
            if not torch.isfinite(t).all():
                raise ValueError(
                    f"{type(self).__name__}: model produced non-finite logits "
                    f"(inf/nan)."
                )

    def preprocess_batch(self, batch: GraphBatch) -> GraphBatch:
        # cast features to f32, targets to f32
        mapping = {
            DataRole.FEATURES: torch.float32,
            DataRole.TARGETS: torch.float32
        }
        return batch.to_dtype(mapping)

    def _compute_out_offsets(self) -> Tensor:
        # the output layout is the same as the input target layout for regression
        decoder = self._ctx.services.require("decode", required_by=type(self))
        layout = decoder.get_segment_layout(DataRole.TARGETS, torch.device("cpu"))
        return torch.tensor(layout.offsets, dtype=torch.long)

    @property
    def use_normalized_targets(self) -> bool:
        return True
