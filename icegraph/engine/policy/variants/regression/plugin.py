# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import ClassVar, Any

import torch

from icegraph.common.data import DataRole

from ...policy import Policy
from ...types import TaskSpec

from .config import RegressionConfig

__all__ = ["Regression"]


class Regression(Policy[RegressionConfig]):
    name: ClassVar[str] = "regression"
    version: ClassVar[int] = 1

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> RegressionConfig:
        return RegressionConfig(**config)

    def _build_task_spec(self) -> TaskSpec:
        # the output layout is the same as the input target layout for regression
        decoder = self._ctx.services.require("decode", required_by=type(self))
        layout = decoder.get_segment_layout(DataRole.TARGETS, torch.device("cpu"))

        return TaskSpec(
            out_offsets=layout.offsets,
            target_dtype=torch.float32,
            norm_targets=False
        )
