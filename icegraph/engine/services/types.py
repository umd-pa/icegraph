# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import dataclass

from icegraph.common.plugins import PluginContext

if TYPE_CHECKING:
    from .manager import ServiceManager

__all__ = ["ServiceContext"]


@dataclass(frozen=True)
class ServiceContext(PluginContext):
    services: "ServiceManager"
    debug: bool
