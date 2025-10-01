# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import List, Optional, Tuple, Union, Callable, Generator, Literal, Dict, Any, TYPE_CHECKING
from abc import abstractmethod

import torch
from torch_geometric.data import Batch

from icegraph.console import Console
from icegraph.utils import Statistics
from icegraph.trainer.callbacks.callback import Callback
from icegraph.types import ComputedMetrics

if TYPE_CHECKING:
    from icegraph.trainer.core.trainer import Trainer
else:
    Trainer = None

__all__ = ["Normalizer"]


class _StatMixin:

    @staticmethod
    def _get_global_stats(trainer: Trainer) -> Tuple[Statistics, Statistics]:
        def iter_shard_stats() -> Generator[Tuple[Statistics, Statistics], Any, None]:
            attrs = trainer.registry.attrs
            try:
                for shard_id, stat_dict in attrs.items():
                    stats = stat_dict["stat"]
                    f_stats_dict, l_stats_dict = stats["feature_stats"]["train"], stats["label_stats"]["train"]
                    yield Statistics.from_dict(f_stats_dict), Statistics.from_dict(l_stats_dict)
            except KeyError as e:
                Console.out(f"Skipping shard {str(shard_id)}: no stats found ({e}).", severity=2)

        trainer.console.log("Collecting dataset global attributes and computing global statistics...")

        global_f: Optional[Statistics] = None
        global_l: Optional[Statistics] = None

        for f_stat, l_stat in iter_shard_stats():
            global_f = f_stat if global_f is None else global_f.merge(f_stat)
            global_l = l_stat if global_l is None else global_l.merge(l_stat)

        if global_f is None or global_l is None:
            raise RuntimeError("No shard statistics found; cannot compute globals.")

        return global_f, global_l


class Normalizer(Callback, torch.nn.Module, _StatMixin):

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
        self.dispatch(batch, trainer)

    def on_batch_end(self, trainer: Trainer, batch: Batch, out: torch.Tensor, target: torch.Tensor, loss: Union[int, float], metrics: ComputedMetrics) -> None:
        if not trainer.model.training:
            self.dispatch(out, trainer, inverse=True)
            self.dispatch(target, trainer, inverse=True)

    def dispatch(self, data: Union[torch.Tensor, Batch], trainer: Optional[Trainer] = None, inverse: bool = False) -> Optional[torch.Tensor]:
        """
        Executes the calculation. Detects if in training or inference mode and dispatches to the evaluator.
        """
        # determine which transform to perform
        operate: Callable[
            [torch.Tensor, Literal['x', 'y']], torch.Tensor
        ] = self.normalize if not inverse else self.inverse_normalize

        if isinstance(data, Batch):
            if trainer is None:
                raise ValueError("Trainer must be provided when normalizing a Batch in training mode.")
            self._ensure_on_device(trainer.device)
            if hasattr(data, "x"):
                data.x = operate(data.x, field='x')

            if hasattr(data, "y"):
                if trainer.strategy.task != "regression":
                    return  # no op
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