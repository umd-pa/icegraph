# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from icegraph.common.plugins import PluginContext

if TYPE_CHECKING:
    from ..services import ServiceManager
    from ..status import Status

__all__ = ["PolicyContext", "TaskSpec"]


@dataclass(frozen=True)
class PolicyContext(PluginContext):
    services: ServiceManager
    status: Status


@dataclass(frozen=True)
class TaskSpec:
    out_offsets: Tensor
    target_dtype: torch.dtype
    norm_targets: bool
