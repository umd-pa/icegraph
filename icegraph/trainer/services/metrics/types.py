# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from dataclasses import dataclass

__all__ = ["ComputedMetric"]


### COMPUTED METRIC

@dataclass
class ComputedMetric:
    name:   str
    value:  float
    ema:    float | None
    delta:  float | None
    span:   int
