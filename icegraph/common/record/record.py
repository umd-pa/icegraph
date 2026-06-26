# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Any
from dataclasses import dataclass

__all__ = ["Record"]


@dataclass(frozen=True)
class Record:
    # index and shard id of the sample
    index: int
    shard_id: str

    # sample dict
    data: dict[str, Any]
