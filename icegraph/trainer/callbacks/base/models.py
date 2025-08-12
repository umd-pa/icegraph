# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Union, Optional
import json

from torch_geometric.data import Batch

from icegraph.data.readers import LMDBConfiguredShardReader, LMDBReader
from icegraph.utils import Statistics
from icegraph.console import Console

if TYPE_CHECKING:
    from icegraph.trainer import Trainer
else:
    class Trainer:
        class Metrics:
            ...

__all__ = ["Callback", "NormCallback"]


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

    def on_train_begin(self, trainer: Trainer) -> None:
        """
        Called before the first training epoch starts.
        Use this to set up any state needed at the very start of training.

        Args:
            trainer (Trainer): The trainer object.
        """
        pass

    def on_train_end(self, trainer: Trainer) -> None:
        """
        Called after the last training epoch finishes.
        Good for final cleanup or summary logging.

        Args:
            trainer (Trainer): The trainer object.
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

    def on_epoch_end(self, trainer: Trainer, epoch: int, metrics: Trainer.Metrics) -> None:
        """
        Called at the end of each epoch, after training (and optional testing)
        for that epoch has completed.

        Args:
            trainer (Trainer): The trainer object.
            epoch: Zero-based index of the epoch that just finished.
            metrics: Training Metrics (SSE and sample count) for that epoch.
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

    def on_batch_end(self, trainer: Trainer, batch: Batch, loss: Union[int, float], metrics: Trainer.Metrics) -> None:
        """
        Called immediately after each batch is processed.

        Args:
            trainer (Trainer): The trainer object.
            batch: The PyG Batch that was just processed.
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

    def on_validation_end(self, trainer: Trainer, epoch: int, metrics: Trainer.Metrics) -> None:
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

    def on_test_end(self, trainer: Trainer, epoch: int, metrics: Trainer.Metrics) -> None:
        """
        Called after testing completes for a given epoch.

        Args:
            trainer (Trainer): The trainer object.
            epoch: Zero-based index of the epoch tested.
            metrics: Test Metrics (SSE and sample count).
        """
        pass

    def on_save(self, trainer: Trainer, epoch: int, metrics: Trainer.Metrics) -> None:
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


class NormCallback(Callback):

    def __init__(self) -> None:
        """Initialize the normalizer."""
        self.f_stats: Optional[Statistics] = None
        self.t_stats: Optional[Statistics] = None

    def on_init(self, trainer: Trainer) -> None:
        # Build global stats once
        map_df = LMDBReader(trainer.datasets.map_file).to_pandas()
        LMDBConfiguredShardReader.configure(trainer.datasets.source, max_open_envs=4, map_df=map_df)
        with LMDBConfiguredShardReader() as reader:
            self.f_stats, self.t_stats = reader.stats  # tuple[Statistics, Statistics]

        # save the params for future use
        self._save_global_stats(trainer)

    def on_batch_transfer(self, trainer: Trainer, batch: Batch) -> None:
        # normalization will always be called on batch transfer so processing can be done on the accelerator
        self._normalize_inplace(trainer, batch)

    def _save_global_stats(self, trainer: Trainer) -> None:
        """Save the normalizer params to disk for renormalization in production."""
        outfile = trainer.outdir / "global_stats.json"
        payload = {
            "f_stats": self.f_stats.to_dict(strip_np=True),
            "t_stats": self.t_stats.to_dict(strip_np=True)
        }
        with outfile.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        Console.out(f"Saved global stats to {outfile}")

    @abstractmethod
    def _normalize_inplace(self, trainer: Trainer, batch: Batch) -> None:
        """Place appropriate normalization code here."""
        ...
