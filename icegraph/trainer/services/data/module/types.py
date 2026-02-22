# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import TYPE_CHECKING

from dataclasses import dataclass

from icegraph.types.data import Split
from icegraph.types.plugins import PluginContext

if TYPE_CHECKING:
    from icegraph.trainer.services.data.store import Store

__all__ = ["ModuleContext"]


@dataclass(frozen=True)
class ModuleContext(PluginContext):
    split: Split
    store: Store
