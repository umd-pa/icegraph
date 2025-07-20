# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from abc import ABC
from typing import TYPE_CHECKING, Union

from torch_geometric.data import Batch

if TYPE_CHECKING:
    from icegraph.trainer import Trainer
else:
    class Trainer:
        class Metrics:
            ...

__all__ = ["Callback"]


class Callback(ABC):
    """
    Abstract base class defining hooks into the Trainer lifecycle.
    """

    def on_init(self, trainer: Trainer):
        """
        Called once, at the end of Trainer.__init__, after
        the model, optimizer, and all attributes are set up.
        """
        pass

    def on_train_begin(self, trainer: Trainer):
        """
        Called before the first training epoch starts.
        Use this to set up any state needed at the very start of training.
        """
        pass

    def on_train_end(self, trainer: Trainer):
        """
        Called after the last training epoch finishes.
        Good for final cleanup or summary logging.
        """
        pass

    def on_epoch_begin(self, trainer: Trainer, epoch: int):
        """
        Called at the start of each epoch.

        Args:
            epoch: Zero-based index of the epoch about to run.
        """
        pass

    def on_epoch_end(self, trainer: Trainer, epoch: int, metrics: Trainer.Metrics):
        """
        Called at the end of each epoch, after training (and optional testing)
        for that epoch has completed.

        Args:
            epoch: Zero-based index of the epoch that just finished.
            metrics: Training Metrics (SSE and sample count) for that epoch.
        """
        pass

    def on_batch_begin(self, trainer: Trainer, batch: Batch):
        """
        Called immediately before each batch is processed in training or eval.

        Args:
            batch: The current PyG Batch instance about to be forwarded.
        """
        pass

    def on_batch_end(self, trainer: Trainer, batch: Batch, loss: Union[int, float], metrics: Trainer.Metrics):
        """
        Called immediately after each batch is processed.

        Args:
            batch: The PyG Batch that was just processed.
            loss: The scalar loss for that batch (detached).
            metrics: Running Metrics object updated with this batch.
        """
        pass

    def on_validation_begin(self, trainer: Trainer, epoch: int):
        """
        Called before running the full validation loop for a given epoch.

        Args:
            epoch: Zero-based index of the epoch at which validation is starting.
        """
        pass

    def on_validation_end(self, trainer: Trainer, epoch: int, metrics: Trainer.Metrics):
        """
        Called after validation completes for a given epoch.

        Args:
            epoch: Zero-based index of the epoch validated.
            metrics: Validation Metrics (SSE and sample count).
        """
        pass

    def on_test_begin(self, trainer: Trainer, epoch: int):
        """
        Called before running the full test loop for a given epoch.

        Args:
            epoch: Zero-based index of the epoch at which testing is starting.
        """
        pass

    def on_test_end(self, trainer: Trainer, epoch: int, metrics: Trainer.Metrics):
        """
        Called after testing completes for a given epoch.

        Args:
            epoch: Zero-based index of the epoch tested.
            metrics: Test Metrics (SSE and sample count).
        """
        pass

    def on_save(self, trainer: Trainer, epoch: int, metrics: Trainer.Metrics):
        """
        Called whenever the Trainer invokes save(), regardless of whether
        it’s a “latest” or “best” checkpoint.

        Args:
            epoch: Zero-based index of the epoch at which save was called.
            metrics: Metrics corresponding to that epoch.
        """
        pass

    def on_teardown(self, trainer: Trainer):
        """
        Called at the very end of Trainer.run(), after train, validate, and
        any final cleanup are complete.
        """
        pass