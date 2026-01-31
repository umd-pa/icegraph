# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

import torch
from torch import Tensor

from icegraph.types.transforms import TransformSpace
from icegraph.types.statistics import StatisticKind
from icegraph.statistics import StatisticService

from ..normalizer import Normalizer

__all__ = ["Centering"]


class Centering(Normalizer):
    """MeanCentering normalizer for shifting input features and target labels to center at the mean."""
    name: str = "centering"

    def _build_offset(self, stats: StatisticService, space: TransformSpace, base: int) -> Tensor:
        # offset is simply the mean
        mean = torch.as_tensor(
            stats.get(StatisticKind.MEAN, space=space, base=base),
            dtype=torch.float32
        )
        return mean

    def _build_scale(self, stats: StatisticService, space: TransformSpace, base: int) -> Tensor:
        # scale factor is 1 (no scaling)
        ones = torch.as_tensor(
            [1] * len(stats.columns),
            dtype=torch.float32
        )
        return ones
