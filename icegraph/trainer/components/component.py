# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import TypeVar, Generic, ClassVar
from abc import ABC

from torch.nn import Module

from icegraph.types.plugins import Plugin

from .types import ComponentContext

__all__ = ["Component"]


C = TypeVar("C")
X = TypeVar("X", bound=ComponentContext)


class Component(Plugin[C, X], Module, ABC, Generic[C, X]):
    compatible: ClassVar[tuple[str, ...]] = tuple()
