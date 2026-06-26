# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from dataclasses import dataclass

from icegraph.common.plugins import PluginContext

__all__ = ["FluxModelContext"]


@dataclass(frozen=True)
class FluxModelContext(PluginContext):
    pass
