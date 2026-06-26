# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from icegraph.common.plugins import PluginContext

from ..services import ServiceManager

__all__ = ["PolicyContext", "TaskSpec"]


@dataclass(frozen=True)
class PolicyContext(PluginContext):
    services: ServiceManager


@dataclass(frozen=True)
class TaskSpec:
    out_offsets: Tensor
    target_dtype: torch.dtype
    norm_targets: bool
