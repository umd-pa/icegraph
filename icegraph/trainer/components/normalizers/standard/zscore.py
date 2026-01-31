# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

import torch
from torch import Tensor

from icegraph.types.transforms import TransformSpace
from icegraph.types.statistics import StatisticKind
from icegraph.statistics import StatisticService

from ..normalizer import Normalizer

__all__ = ["ZScore"]


class ZScore(Normalizer):
    """Z-score normalizer."""
    name: str = "zscore"

    def _build_offset(self, stats: StatisticService, space: TransformSpace, base: int) -> Tensor:
        # offset is simply the mean
        mean = torch.as_tensor(
            stats.get(StatisticKind.MEAN, space=space, base=base),
            dtype=torch.float32
        )
        return mean

    def _build_scale(self, stats: StatisticService, space: TransformSpace, base: int) -> Tensor:
        # scale is reciprocal of std
        std = torch.as_tensor(
            stats.std(space=space, base=base),
            dtype=torch.float32
        )
        return std.clamp_min(1e-12).reciprocal()
