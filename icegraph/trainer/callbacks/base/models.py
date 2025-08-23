# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Union, Optional, List, Literal, Dict, Generator, Tuple, Any, Callable, Self

from torch_geometric.data import Batch
import torch

from icegraph.data.readers import LMDBDatasetShardReader, LMDBReader
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

    def on_batch_end(self, trainer: Trainer, batch: Batch, out: torch.Tensor, target: torch.Tensor, loss: Union[int, float], metrics: Trainer.Metrics) -> None:
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


class StatMixin:

    @staticmethod
    def _get_global_stats(trainer: Trainer) -> Tuple[Statistics, Statistics]:
        def iter_shard_stats() -> Generator[Tuple[Statistics, Statistics], Any, None]:
            with LMDBDatasetShardReader() as reader:
                attrs = reader.attrs()
                try:
                    for file_idx, stat_dict in attrs.items():
                        stats = stat_dict["stat"]
                        f_stats_dict, l_stats_dict = stats["feature_stats"], stats["label_stats"]
                        yield Statistics.from_dict(f_stats_dict), Statistics.from_dict(l_stats_dict)
                except KeyError as e:
                    Console.out(f"Skipping file_idx {file_idx}: no stats found ({e}).", severity=2)

        Console.out("Collecting dataset global attributes and computing global statistics...")

        global_f: Optional[Statistics] = None
        global_l: Optional[Statistics] = None

        for f_stat, l_stat in iter_shard_stats():
            global_f = f_stat if global_f is None else global_f.merge(f_stat)
            global_l = l_stat if global_l is None else global_l.merge(l_stat)

        if global_f is None or global_l is None:
            raise RuntimeError("No shard statistics found; cannot compute globals.")

        return global_f, global_l


class Normalizer(Callback, torch.nn.Module, StatMixin):

    def __init__(self, param_list: List[str], **kwargs) -> None:
        """Initialize the normalizer."""
        super().__init__()

        self.f_stats: Optional[Statistics] = None
        self.l_stats: Optional[Statistics] = None

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

    def on_init(self, trainer: Trainer) -> None:
        # Build global stats once on trainer init
        self.f_stats, self.l_stats = self._get_global_stats(trainer)  # tuple[Statistics, Statistics]

        # build params
        self._configure(trainer)

        # register these params
        for param, tensor in self._params.items():
            self.register_buffer(param, tensor, persistent=True)

        self._on_device = False

    def on_batch_transfer(self, trainer: Trainer, batch: Batch) -> None:
        # normalization will always be called on batch transfer so processing can be done on the accelerator
        self._ensure_on_device(trainer.device)
        self.dispatch(batch, trainer)

    def on_batch_end(self, trainer: Trainer, batch: Batch, out: torch.Tensor, target: torch.Tensor, loss: Union[int, float], metrics: Trainer.Metrics) -> None:
        if not trainer.model.training:
            self._ensure_on_device(trainer.device)
            self.dispatch(out, trainer, inverse=True)
            self.dispatch(target, trainer, inverse=True)

    def dispatch(self, data: Union[torch.Tensor, Batch], trainer: Optional[Trainer] = None, inverse: bool = False) -> Optional[torch.Tensor]:
        """
        Executes the calculation. Detects if in training or inference mode and dispatches to the evaluator.

        Returns:
            - torch.Tensor if on inference
            - None if on training
        """
        # determine which transform to perform
        operate: Callable[[torch.Tensor, Literal['x', 'y']], torch.Tensor] = self.normalize if not inverse else self.inverse_normalize

        if isinstance(data, Batch):
            if trainer is None:
                raise ValueError("Trainer must be provided when normalizing a Batch in training mode.")
            self._ensure_on_device(trainer.device)
            if hasattr(data, "x"):
                data.x = operate(data.x, field='x')

            if hasattr(data, "y"):
                data.y = operate(data.y, field="y")

        elif isinstance(data, torch.Tensor):
            self._ensure_on_device(data.device)
            return operate(data, field='y')

        else:
            raise TypeError(f"Unsupported input type {type(data)}")

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

    @abstractmethod
    def inverse_normalize(self, tensor: torch.Tensor, field: Literal['x', 'y']) -> torch.Tensor:
        """
        Apply inverse normalization to a tensor.

        Args:
            tensor (Tensor): Feature or label tensor.
            field (Literal['x', 'y']): Whether this tensor represents features or labels.

        Returns:
            Tensor: De-normalized tensor (same shape).
        """
        ...
