# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, TypeVar, Generic, Any

from .context import InitContext

__all__ = ["Callback"]

if TYPE_CHECKING:
    from ..engine import Engine

E = TypeVar("E", bound="Engine[Any]")


class Callback(ABC, Generic[E]):
    """Hooks into the Engine lifecycle."""

    def on_init(self, ctx: InitContext[E]) -> None:
        """
        Called once, at the end of __init__.

        Args:
            ctx (context.InitContext): Initialization context.
        """
        pass
