# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import TypeAlias

from torch import Tensor

__all__ = ["HistogramState"]


# dense int64 counts of shape [*bins], or None before any rows are seen (the identity)
HistogramState: TypeAlias = "Tensor | None"
