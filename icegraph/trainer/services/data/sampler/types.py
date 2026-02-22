# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from dataclasses import dataclass

from icegraph.types.plugins import PluginContext

from ..types import SizedDataset

__all__ = ["SamplerContext"]


@dataclass(frozen=True)
class SamplerContext(PluginContext):
    dataset:        SizedDataset
    num_replicas:   int
    rank:           int
    shuffle:        bool            = False
