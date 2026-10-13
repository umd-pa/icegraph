# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Any, ClassVar

import torch
from torch import Tensor

from icegraph.common.tensors import SegmentedTensor

from ...plugin import KNNEdgeBuilder

from .config import GaussianConfig

__all__ = ["Gaussian"]


class Gaussian(KNNEdgeBuilder[GaussianConfig]):
    """Weight each edge using a gaussian."""
    name: ClassVar[str] = "gaussian"
    version: ClassVar[int] = 1

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> GaussianConfig:
        return GaussianConfig(**config)

    def build_attr(self, t: SegmentedTensor, edge_index: Tensor) -> Tensor:
        values = t.block(self.config.weight_cols)

        # squared distance across each edge
        difference = values[edge_index[0]] - values[edge_index[1]]
        squared = difference.square().sum(dim=1, keepdim=True)

        return torch.exp(squared / (-2 * self.config.sigma ** 2))
