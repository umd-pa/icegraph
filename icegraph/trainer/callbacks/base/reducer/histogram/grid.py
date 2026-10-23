# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

import dataclasses

from dataclasses import dataclass
from typing import Self

import torch
from torch import Tensor

__all__ = ["Grid"]


@dataclass(frozen=True)
class Grid:
    """Per-label binning in transformed space. Bin i spans ``mins + i / scale``."""
    mins:   Tensor
    maxs:   Tensor
    scale:  Tensor  # bins / (maxs - mins)

    def to(self, device: torch.device | str) -> Self:
        return dataclasses.replace(
            self, mins=self.mins.to(device), maxs=self.maxs.to(device), scale=self.scale.to(device)
        )
