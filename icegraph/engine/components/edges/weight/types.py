# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from dataclasses import dataclass

from icegraph.common.plugins import PluginContext

__all__ = ["EdgeWeightContext"]


@dataclass(frozen=True)
class EdgeWeightContext(PluginContext):
    pass
