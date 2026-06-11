# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import dataclass

from icegraph.common.plugins import PluginContext

from ..types import SizedDataset

if TYPE_CHECKING:
    from torch import device

__all__ = ["SamplerContext"]


@dataclass(frozen=True)
class SamplerContext(PluginContext):
    dataset:        SizedDataset
    num_replicas:   int
    rank:           int
    device:         device
    shuffle:        bool            = False
