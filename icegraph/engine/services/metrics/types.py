# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

__all__ = ["MetricValue", "HeadValues"]


### COMPUTED METRIC

@dataclass
class MetricValue:
    repr:       str
    value:      HeadValues
    ema:        HeadValues
    delta:      HeadValues
    span:       int
    optimum:    float


HeadValues = tuple[Tensor | None, ...]
