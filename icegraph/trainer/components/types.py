# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from dataclasses import dataclass

from icegraph.types.plugins import PluginContext

from ..services import ServiceManager

__all__ = ["ComponentContext"]


@dataclass(frozen=True)
class ComponentContext(PluginContext):
    services: ServiceManager
