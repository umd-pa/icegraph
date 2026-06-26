# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar, Generic, Any

if TYPE_CHECKING:
    from icegraph.engine import Engine

__all__ = ["Context", "InitContext"]


E = TypeVar("E", bound="Engine[Any]")


# BASE CONTEXT
@dataclass(frozen=True, slots=True)
class Context(Generic[E]):
    engine: E

@dataclass(frozen=True, slots=True)
class InitContext(Context[E]):
    ...
