# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from dataclasses import dataclass

from icegraph.types.plugins import PluginContext
from icegraph.types.files import Source

__all__ = ["StoreContext"]


@dataclass(frozen=True)
class StoreContext(PluginContext):
    source: Source
