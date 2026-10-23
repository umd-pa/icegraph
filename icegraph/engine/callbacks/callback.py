# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, TypeVar, Generic, Any

from .context import InitContext, StatusContext

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

    def on_status(self, ctx: StatusContext[E]) -> None:
        """
        Called for every event the engine's status reports: a ``TaskEvent`` when a task
        starts, progresses, finishes or fails, and a ``LogEvent`` for each record logged.
        The task on an event is live, and the full picture (running tasks, recent logs,
        throughputs) is on ``ctx.engine.status``.

        Fires during setup, before services and components exist. May fire from any thread.

        Args:
            ctx (context.StatusContext): Status context.
        """
        pass
