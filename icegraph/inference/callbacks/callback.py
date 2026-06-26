# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

from icegraph.engine.callbacks import Callback

from . import context

if TYPE_CHECKING:
    from ..inference import BatchInference

__all__ = ["InferenceCallback"]


class InferenceCallback(Callback["BatchInference"], ABC):
    """Hooks into the Inference lifecycle."""

    def on_execute(self, ctx: context.ExecuteContext) -> None:
        """
        Called once, before execution begins.

        Args:
            ctx (context.ExecuteContext): Execution context.
        """
        pass

    def on_teardown(self, ctx: context.TeardownContext) -> None:
        """
        Called once, during teardown.

        Args:
            ctx (context.TeardownContext): Teardown context.
        """
        pass
