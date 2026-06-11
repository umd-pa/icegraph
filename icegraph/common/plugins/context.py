# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["PluginContext"]


@dataclass(frozen=True)
class PluginContext:
    ...