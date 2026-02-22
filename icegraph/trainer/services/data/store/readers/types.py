# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from icegraph.types.plugins import PluginContext


@dataclass(frozen=True)
class ReaderContext(PluginContext):
    path: Path
