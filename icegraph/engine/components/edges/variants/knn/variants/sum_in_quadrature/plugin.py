# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Any, ClassVar

import torch
from torch import Tensor

from icegraph.common.tensors import SegmentedTensor

from ...plugin import KNNEdgeBuilder

from .config import SumInQuadratureConfig

__all__ = ["SumInQuadrature"]


class SumInQuadrature(KNNEdgeBuilder[SumInQuadratureConfig]):
    """Weight each edge by the quadrature sum of its endpoint differences."""
    name: ClassVar[str] = "sum-in-quadrature"
    version: ClassVar[int] = 1

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> SumInQuadratureConfig:
        return SumInQuadratureConfig(**config)

    def build_attr(self, t: SegmentedTensor, edge_index: Tensor) -> Tensor:
        # gathering by edge materializes anyway, so a view is enough here
        values = t.block(self.config.weight_cols)

        # difference across each edge, then the euclidean norm over the selected columns
        difference = values[edge_index[0]] - values[edge_index[1]]
        return torch.linalg.vector_norm(difference, dim=1, keepdim=True)
