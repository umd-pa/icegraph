# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from abc import ABC

from . import context

__all__ = ["Callback"]


class Callback(ABC):
    """
    Abstract base class defining hooks into the Trainer lifecycle.
    """

    def on_init(self, ctx: context.InitContext) -> None:
        """
        Called once, at the end of Inference.__init__, after
        the model and all attributes are set up.

        Args:
            ctx (context.InitContext): Initialization context.
        """
        pass

    def on_execute(self, ctx: context.ExecuteContext) -> None:
        """
        Called once, before execution begins.

        Args:
            ctx (context.ExecuteContext): Execution context.
        """
        pass

    def on_batch_begin(self, ctx: context.BatchBeginContext) -> None:
        """
        Called immediately before each batch is processed.

        Args:
            ctx (context.BatchBeginContext): Batch-begin context (includes the current PyG Batch).
        """
        pass

    def on_batch_transfer(self, ctx: context.BatchTransferContext) -> None:
        """
        Called immediately after batch has been transferred to device.

        Args:
            ctx (context.BatchTransferContext): Batch-transfer context (includes the current PyG Batch).
        """

    def on_batch_end(self, ctx: context.BatchEndContext) -> None:
        """
        Called immediately after each batch is processed.

        Args:
            ctx (context.BatchEndContext): Batch-end context (includes batch, out, target, and loss).
        """
        pass

    def on_teardown(self, ctx: context.TeardownContext) -> None:
        """
        Called at the very end of Trainer.run(), after train, validate, and
        any final cleanup are complete.

        Args:
            ctx (context.TeardownContext): Teardown context.
        """
        pass
