# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import ClassVar

from torch import Tensor

from ...plugin import RegressionMetric
from ...config import Config

__all__ = ["MAE"]


class MAE(RegressionMetric[Config]):
    """Per-head mean absolute error over segmented predictions."""
    name: ClassVar[str] = "mae"
    version: ClassVar[int] = 1

    def repr(self) -> str:
        return "mae"

    def residual(self, diff: Tensor) -> Tensor:
        # |out - target|
        return diff.abs_()
