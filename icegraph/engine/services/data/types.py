# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import TypeVar, Protocol
from collections.abc import Sized

__all__ = ["SizedDataset"]


### PROTOCOLS

D = TypeVar("D")

class SizedDataset(Protocol[D], Sized):
    def __getitem__(self, index: int | slice) -> D: ...
    def __len__(self) -> int: ...
