# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from icegraph.common.tensors import DualResidentTensor

__all__ = ["DualResidentBounds"]


@dataclass(frozen=True)
class DualResidentBounds:
    mins: DualResidentTensor
    maxs: DualResidentTensor

    def on(self, device: torch.device | str) -> tuple[Tensor, Tensor]:
        return self.mins.on(device), self.maxs.on(device)
