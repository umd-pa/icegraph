# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import TypeAlias
from dataclasses import dataclass

import torch
from torch import Tensor

from icegraph.common.tensors import DualResidentTensor

__all__ = ["Binning", "HistogramState"]


# dense int64 counts of shape [*bins], or None before any rows are seen (the identity)
HistogramState: TypeAlias = "Tensor | None"


@dataclass(frozen=True)
class Binning:
    """Per-label binning in transformed space. Bin i spans ``mins + i / scale``."""
    mins:   DualResidentTensor
    maxs:   DualResidentTensor
    scale:  DualResidentTensor  # bins / (maxs - mins)

    def on(self, device: torch.device | str) -> tuple[Tensor, Tensor, Tensor]:
        return self.mins.on(device), self.maxs.on(device), self.scale.on(device)
