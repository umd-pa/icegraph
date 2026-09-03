# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import ClassVar

from torch import Tensor

from ...plugin import RegressionMetric
from ...config import Config

__all__ = ["RMSE"]


class RMSE(RegressionMetric[Config]):
    """Per-head root mean squared error over segmented predictions.

    The accumulator stays in the squared domain because the square root does not
    distribute over addition, thus it is taken once, after the full mean is known.
    """
    name: ClassVar[str] = "rmse"
    version: ClassVar[int] = 1

    def repr(self) -> str:
        return "rmse"

    def residual(self, diff: Tensor) -> Tensor:
        # (out - target) ** 2
        return diff.mul_(diff)

    def resolve(self, mean: Tensor) -> Tensor:
        # root taken once, on the resolved mean
        return mean.sqrt_()
