# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

import torch
from torch import Tensor

from icegraph.types.transforms import TransformSpace
from icegraph.types.statistics import StatisticKind
from icegraph.statistics import StatisticService

from ..normalizer import Normalizer

__all__ = ["MinMax"]


class MinMax(Normalizer):
    """MinMax normalizer for scaling input features and target labels to [0, 1] range."""
    name: str = "minmax"

    def _build_offset(self, stats: StatisticService, space: TransformSpace, base: int) -> Tensor:
        # offset is minimum (shift to 0)
        minimum = torch.as_tensor(
            stats.get(StatisticKind.MIN, space=space, base=base),
            dtype=torch.float32
        )
        return minimum

    def _build_scale(self, stats: StatisticService, space: TransformSpace, base: int) -> Tensor:
        # scale factor is 1/range
        range_ = torch.as_tensor(
            stats.range(space=space, base=base),
            dtype=torch.float32
        )
        return range_.clamp_min(1e-12).reciprocal()
