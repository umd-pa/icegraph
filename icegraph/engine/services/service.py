# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import TypeVar, ClassVar
from abc import ABC

from icegraph.common.plugins import Plugin

from .types import ServiceContext

__all__ = ["Service"]


C = TypeVar("C")


class Service(Plugin[C, ServiceContext], ABC):
    # any service dependencies this service has
    deps: ClassVar[tuple[str, ...]] = tuple()

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()

        if not isinstance(cls.deps, tuple) or not all(isinstance(d, str) for d in cls.deps):
            raise RuntimeError(f"Dependencies for service {cls.__name__} must be a tuple of str.")
