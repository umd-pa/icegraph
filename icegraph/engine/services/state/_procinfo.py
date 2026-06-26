# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ProcInfo"]


@dataclass(frozen=True)
class ProcInfo:
    rank:       int = 0
    world:      int = 1
    local_rank: int = 0
