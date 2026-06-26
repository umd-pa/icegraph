# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from dataclasses import dataclass

from torch import Tensor

__all__ = ["ComputedMetric"]


### COMPUTED METRIC

@dataclass
class ComputedMetric:
    repr:       str
    value:      Tensor
    ema:        Tensor | None
    delta:      Tensor | None
    span:       int
    optimum:    float
