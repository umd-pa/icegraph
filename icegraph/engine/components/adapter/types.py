# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from dataclasses import dataclass

from ..types import ComponentContext

__all__ = ["AdapterContext"]


@dataclass(frozen=True)
class AdapterContext(ComponentContext):
    pass
