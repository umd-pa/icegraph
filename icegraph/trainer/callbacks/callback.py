# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from abc import ABC
from typing import TYPE_CHECKING, Union

from torch_geometric.data import Batch
import torch

from icegraph.types import ComputedMetrics

if TYPE_CHECKING:
    from icegraph.trainer import Trainer
else:
    Trainer = None

__all__ = ["Callback"]


class Callback(ABC):
    """
    Abstract base class defining hooks into the Trainer lifecycle.
    """

    def on_init(self, trainer: Trainer) -> None:
        """
        Called once, at the end of Trainer.__init__, after
        the model, optimizer, and all attributes are set up.

        Args:
            trainer (Trainer): The trainer object.
        """
        pass

    def on_execute(self, trainer: Trainer) -> None:
        """
        Called once, before execution begins.

        Args:
            trainer (Trainer): The trainer object.
        """
        pass

    def on_train_begin(self, trainer: Trainer, epoch: int) -> None:
        """
        Called before the training epoch starts.

        Args:
            trainer (Trainer): The trainer object.
            epoch: Zero-based index of the epoch about to run.
        """
        pass

    def on_train_end(self, trainer: Trainer, epoch: int, metrics: ComputedMetrics) -> None:
        """
        Called after the training epoch finishes.

        Args:
            trainer (Trainer): The trainer object.
            epoch: Zero-based index of the epoch about to run.
            metrics: Training Metrics (SSE and sample count) for that epoch.
        """
        pass

    def on_epoch_begin(self, trainer: Trainer, epoch: int) -> None:
        """
        Called at the start of each epoch.

        Args:
            trainer (Trainer): The trainer object.
            epoch: Zero-based index of the epoch about to run.
        """
        pass

    def on_epoch_end(self, trainer: Trainer, epoch: int) -> None:
        """
        Called at the end of each epoch, after training (and optional testing)
        for that epoch has completed.

        Args:
            trainer (Trainer): The trainer object.
            epoch: Zero-based index of the epoch that just finished.
        """
        pass

    def on_batch_begin(self, trainer: Trainer, batch: Batch) -> None:
        """
        Called immediately before each batch is processed in training or eval.

        Args:
            trainer (Trainer): The trainer object.
            batch: The current PyG Batch instance about to be forwarded.
        """
        pass

    def on_batch_transfer(self, trainer: Trainer, batch: Batch) -> None:
        """
        Called immediately after batch has been transferred to GPU.

        Args:
            trainer (Trainer): The trainer object.
            batch: The current PyG Batch instance about to be forwarded.
        """

    def on_batch_end(self, trainer: Trainer, batch: Batch, out: torch.Tensor, target: torch.Tensor, loss: Union[int, float], metrics: ComputedMetrics) -> None:
        """
        Called immediately after each batch is processed.

        Args:
            trainer (Trainer): The trainer object.
            batch: The PyG Batch that was just processed.
            out: Tensor of predicted values.
            target: Tensor of target values.
            loss: The scalar loss for that batch (detached).
            metrics: Running Metrics object updated with this batch.
        """
        pass

    def on_validation_begin(self, trainer: Trainer, epoch: int) -> None:
        """
        Called before running the full validation loop for a given epoch.

        Args:
            trainer (Trainer): The trainer object.
            epoch: Zero-based index of the epoch at which validation is starting.
        """
        pass

    def on_validation_end(self, trainer: Trainer, epoch: int, metrics: ComputedMetrics) -> None:
        """
        Called after validation completes for a given epoch.

        Args:
            trainer (Trainer): The trainer object.
            epoch: Zero-based index of the epoch validated.
            metrics: Validation Metrics (SSE and sample count).
        """
        pass

    def on_test_begin(self, trainer: Trainer, epoch: int) -> None:
        """
        Called before running the full test loop for a given epoch.

        Args:
            trainer (Trainer): The trainer object.
            epoch: Zero-based index of the epoch at which testing is starting.
        """
        pass

    def on_test_end(self, trainer: Trainer, epoch: int, metrics: ComputedMetrics) -> None:
        """
        Called after testing completes for a given epoch.

        Args:
            trainer (Trainer): The trainer object.
            epoch: Zero-based index of the epoch tested.
            metrics: Test Metrics (SSE and sample count).
        """
        pass

    def on_save(self, trainer: Trainer, epoch: int, metrics: ComputedMetrics) -> None:
        """
        Called whenever the Trainer invokes save(), regardless of whether
        it’s a “latest” or “best” checkpoint.

        Args:
            trainer (Trainer): The trainer object.
            epoch: Zero-based index of the epoch at which save was called.
            metrics: Metrics corresponding to that epoch.
        """
        pass

    def on_teardown(self, trainer: Trainer) -> None:
        """
        Called at the very end of Trainer.run(), after train, validate, and
        any final cleanup are complete.

        Args:
            trainer (Trainer): The trainer object.
        """
        pass
