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
        Called once, at the end of Trainer.__init__, after
        the model, optimizer, and all attributes are set up.

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

    def on_epoch_begin(self, ctx: context.EpochBeginContext) -> None:
        """
        Called at the start of each epoch.

        Args:
            ctx (context.EpochBeginContext): Epoch-begin context.
        """
        pass

    def on_epoch_end(self, ctx: context.EpochEndContext) -> None:
        """
        Called at the end of each epoch, after training (and optional testing)
        for that epoch has completed.

        Args:
            ctx (context.EpochEndContext): Epoch-end context.
        """
        pass

    def on_batch_begin(self, ctx: context.BatchBeginContext) -> None:
        """
        Called immediately before each batch is processed in training or eval.

        Args:
            ctx (context.BatchBeginContext): Batch-begin context (includes the current PyG Batch).
        """
        pass

    def on_batch_end(self, ctx: context.BatchEndContext) -> None:
        """
        Called immediately after each batch is processed.

        Args:
            ctx (context.BatchEndContext): Batch-end context (includes batch, out, target, and loss).
        """
        pass

    def on_train_begin(self, ctx: context.TrainBeginContext) -> None:
        """
        Called before the training epoch starts.

        Args:
            ctx (context.TrainBeginContext): Train-begin context.
        """
        pass

    def on_train_end(self, ctx: context.TrainEndContext) -> None:
        """
        Called after the training epoch finishes.

        Args:
            ctx (context.TrainEndContext): Train-end context (includes epoch loss).
        """
        pass

    def on_validation_begin(self, ctx: context.ValidationBeginContext) -> None:
        """
        Called before running the full validation loop for a given epoch.

        Args:
            ctx (context.ValidationBeginContext): Validation-begin context.
        """
        pass

    def on_validation_end(self, ctx: context.ValidationEndContext) -> None:
        """
        Called after validation completes for a given epoch.

        Args:
            ctx (context.ValidationEndContext): Validation-end context (includes epoch loss).
        """
        pass

    def on_test_begin(self, ctx: context.TestBeginContext) -> None:
        """
        Called before running the full test loop for a given epoch.

        Args:
            ctx (context.TestBeginContext): Test-begin context.
        """
        pass

    def on_test_end(self, ctx: context.TestEndContext) -> None:
        """
        Called after testing completes for a given epoch.

        Args:
            ctx (context.TestEndContext): Test-end context (includes epoch loss).
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
