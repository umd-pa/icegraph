# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from dataclasses import dataclass

from icegraph.types.plugins import PluginContext
from icegraph.trainer.services.data import DataView

__all__ = ["StrategyContext"]


@dataclass(frozen=True)
class StrategyContext(PluginContext):
    data: DataView
