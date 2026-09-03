# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import ClassVar

from torch import Tensor

from ...plugin import RegressionMetric
from ...config import Config

__all__ = ["MSE"]


class MSE(RegressionMetric[Config]):
    """Per-head mean squared error over segmented predictions."""
    name: ClassVar[str] = "mse"
    version: ClassVar[int] = 1

    def repr(self) -> str:
        return "mse"

    def residual(self, diff: Tensor) -> Tensor:
        # (out - target) ** 2
        return diff.mul_(diff)
