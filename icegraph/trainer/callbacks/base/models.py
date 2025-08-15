# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Union, Optional, List, Literal, Dict

from torch_geometric.data import Batch
import torch

from icegraph.data.readers import LMDBConfiguredShardReader, LMDBReader
from icegraph.utils import Statistics
from icegraph.console import Console

if TYPE_CHECKING:
    from icegraph.trainer import Trainer
else:
    class Trainer:
        class Metrics:
            ...

__all__ = ["Callback", "Normalizer"]


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


class Normalizer(Callback, torch.nn.Module):

    def __init__(self, param_list: List[str], **kwargs) -> None:
        """Initialize the normalizer."""
        super().__init__()

        self.f_stats: Optional[Statistics] = None
        self.t_stats: Optional[Statistics] = None

        # on device flag
        self._on_device: bool = False

        # eps for use in div by zero cases
        self._eps: float = 1e-8

        # build the params dict
        self._params: Dict[str, Optional[torch.Tensor]] = {param: kwargs.get(param, None) for param in param_list}

        # ensure that if one param is passed, all are passed
        param_mask = [param is not None for param in self._params.values()]
        if not all(param_mask) and any(param_mask):
            raise ValueError(f"Must pass no parameters or all parameters to {self.__class__.__name__}.")

        # register these params
        for param, tensor in self._params.items():
            self.register_buffer(param, tensor, persistent=True)

    def on_init(self, trainer: Trainer) -> None:
        # Build global stats once on trainer init
        map_df = LMDBReader(trainer.datasets.map_file).to_pandas().sort_values(by="index").reset_index(drop=True)
        LMDBConfiguredShardReader.configure(trainer.datasets.source, max_open_envs=4, map_df=map_df)
        with LMDBConfiguredShardReader() as reader:
            self.f_stats, self.t_stats = reader.stats  # tuple[Statistics, Statistics]

        # build params
        self._configure(trainer)
        self._on_device = False

        # save params for inference
        self.save(trainer)

    def on_batch_transfer(self, trainer: Trainer, batch: Batch) -> None:
        # normalization will always be called on batch transfer so processing can be done on the accelerator
        self._ensure_on_device(trainer.device)
        self.dispatch(batch, trainer)

    def dispatch(self, data: Union[torch.Tensor, Batch], trainer: Optional[Trainer] = None) -> Optional[torch.Tensor]:
        """
        Executes the calculation. Detects if in training or inference mode and dispatches to the evaluator.

        Returns:
            - torch.Tensor if on inference
            - None if on training
        """
        if isinstance(data, Batch):
            if trainer is None:
                raise ValueError("Trainer must be provided when normalizing a Batch in training mode.")
            self._ensure_on_device(trainer.device)
            self._batch_dispatch(data)

        elif isinstance(data, torch.Tensor):
            self._ensure_on_device(data.device)
            return self.normalize(data, field='y')

        else:
            raise TypeError(f"Unsupported input type {type(data)}")

    def save(self, trainer: Trainer):
        """Save the normalizer params to disk for renormalization in production."""
        outfile = trainer.outdir / f"norm_params_{self.__class__.__name__.lower()}.pth"
        torch.save(self.state_dict(), outfile)
        Console.out(f"Saved global stats to {outfile}")

    def load(self, path: str, map_location=None):
        """Load the normalizer params from disk for renormalization in production."""
        self.load_state_dict(torch.load(path, map_location=map_location))

    def _ensure_on_device(self, device: torch.device) -> None:
        """
        Lazily move normalization parameters to the specified device.

        Args:
            device (device): The target device (CPU or GPU) to move normalization parameters onto.
        """
        if self._on_device:
            return

        # move all params to device
        for param, tensor in self._params.items():
            if tensor is not None:
                self._params[param] = tensor.to(device, non_blocking=True)

        self._on_device = True

    def _batch_dispatch(self, batch: Batch) -> None:
        """Normalize a Batch object in-place."""
        if hasattr(batch, "x"):
            batch.x = self.normalize(batch.x, field='x')

        if hasattr(batch, "y"):
            batch.y = self.normalize(batch.y, field="y")

    @abstractmethod
    def _configure(self, trainer: Trainer) -> None:
        """Configure the params for use in normalization."""
        ...

    @abstractmethod
    def normalize(self, tensor: torch.Tensor, field: Literal['x', 'y']) -> torch.Tensor:
        """
        Apply normalization to a tensor.

        Args:
            tensor (Tensor): Feature or label tensor.
            field (Literal['x', 'y']): Whether this tensor represents features or labels.

        Returns:
            Tensor: Normalized tensor (same shape).
        """
        ...
