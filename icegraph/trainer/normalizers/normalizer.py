# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import Optional, Tuple, Union, Callable, Iterator, Literal, Dict, TYPE_CHECKING, Any, TypeAlias
from abc import abstractmethod
import functools

import torch
from torch_geometric.data import Batch
from torch import Tensor

from icegraph.console import Console
from icegraph.utils import Statistics
from icegraph.trainer.callbacks.callback import Callback
from icegraph.types import ComputedMetrics

if TYPE_CHECKING:
    from icegraph.trainer.core.trainer import Trainer
else:
    Trainer = None

__all__ = ["Normalizer", "NormTarget"]


NormTarget: TypeAlias = Literal["features", "labels"]


class Normalizer(Callback, torch.nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the normalizer."""
        super().__init__(*args, **kwargs)

        # grab passed device
        self.device = kwargs.get("device", "cpu")

        # eps for use in div by zero cases
        self._eps: float = 1e-8

        # statistic object caches
        self._x_stats: Statistics
        self._y_stats: Statistics

        # grab passed attributes
        self._attrs: Dict[str, Any] = kwargs.get("attrs", {})

        # computed value cache
        self._cache: Dict[str, Tensor] = {}

    def get_extra_state(self) -> Dict[str, Any]:
        # Must be picklable
        return {
            "_x_stats": self._x_stats,
            "_y_stats": self._y_stats,
            "_attrs": self._attrs
        }

    def set_extra_state(self, state) -> None:
        self._x_stats = state.get("_x_stats")
        self._y_stats = state.get("_y_stats")
        self._attrs = state.get("_attrs")

    def on_init(self, trainer: Trainer) -> None:
        # build params
        self._x_stats, self._y_stats = self._get_global_stats(trainer)

    def on_batch_transfer(self, trainer: Trainer, batch: Batch) -> None:
        # normalization will always be called on batch transfer so processing can be done on the accelerator
        self.dispatch(batch, trainer)

    def on_batch_end(self, trainer: Trainer, batch: Batch, out: Tensor, target: Tensor, loss: Union[int, float], metrics: ComputedMetrics) -> None:
        if not trainer.model.training:
            for tensor in [out, target]:
                self.dispatch(tensor, trainer, inverse=True, target="labels")

    def dispatch(self, data: Union[Tensor, Batch], trainer: Optional[Trainer] = None, inverse: bool = False, target: NormTarget = "features") -> Union[Tensor, Batch]:
        """
        Executes the calculation. Detects if in training or inference mode and dispatches to the evaluator. Normalizes tensors/batches in-place.
        """
        # raise if the input data is not of valid dtype
        if not isinstance(data, (Batch, Tensor)):
            raise TypeError(f"Unsupported input type {type(data)}")

        # grab active task if available
        task: Optional[str] = trainer.strategy.task if trainer is not None else None

        # determine which transform to perform
        transform_fn: Callable[
            [Tensor, NormTarget], Tensor
        ] = self.normalize if not inverse else self.inverse_normalize

        # if the input is a Tensor, not in training mode
        if isinstance(data, Tensor):
            # transform in place
            return transform_fn(data, target)

        # handle batch inputs, only normalize labels if in regression
        if hasattr(data, "x"):
            data.x = transform_fn(data.x, "features")

        if hasattr(data, "y") and task == "regression":
            data.y = transform_fn(data.y, "labels")

        return data

    @staticmethod
    def _iter_shard_stats(trainer: Trainer) -> Iterator[Tuple[Statistics, Statistics]]:
        """Yield (feature_stats, label_stats) for each shard that has stats."""
        for shard_id, stat_dict in trainer.registry.attrs.items():
            try:
                stats = stat_dict["stat"]
                x_stats_dict = stats["feature_stats"]["train"]
                y_stats_dict = stats["label_stats"]["train"]
            except KeyError as e:
                Console.out(
                    f"Skipping shard {shard_id}: no stats found ({e}).",
                    severity=2,
                )
                continue

            yield Statistics.load_struct(x_stats_dict), Statistics.load_struct(y_stats_dict)

    @classmethod
    def _get_global_stats(cls, trainer: Trainer) -> Tuple[Statistics, Statistics]:
        trainer.console.log(
            "Collecting dataset attributes and computing global statistics..."
        )

        shard_iter = cls._iter_shard_stats(trainer)  # or whatever class this lives in

        # prime the accumulator with the first shard
        try:
            global_x, global_y = next(shard_iter)
        except StopIteration:
            raise RuntimeError("No shard statistics found; cannot compute globals.")

        # merge the rest
        for x_stat, y_stat in shard_iter:
            global_x += x_stat
            global_y += y_stat

        return global_x, global_y

    def _to_device(self, tensor: Tensor) -> Tensor:
        return tensor.to(self.device, non_blocking=True)

    def _cached(self, build: Callable[[], Tensor], key: str, dtype: torch.dtype = torch.float32) -> Tensor:
        """Executes the function and caches the results, then returns the result."""
        if key not in self._cache:
            self._cache[key] = self._to_device(torch.as_tensor(build(), dtype=dtype))

        return self._cache[key]

    @staticmethod
    def _with_tensor_format(
            func: Callable[[Any, Tensor, NormTarget], Tensor]
    ) -> Callable[[Any, Tensor, NormTarget], Tensor]:
        """
        Decorator to wrap any normalization methods, helps to ensure
        standardized tensor formatting before and after normalization.
        """
        @functools.wraps(func)
        def inner(self, tensor: Tensor, target: NormTarget) -> Tensor:
            # make sure input is a float tensor
            if not torch.is_floating_point(tensor):
                tensor = tensor.float()

            # unsqueeze any 1D tensors so dimensions match
            if unsqueezed := tensor.ndim == 1:
                tensor = tensor.unsqueeze(1)

            tensor = func(self, tensor, target)

            # resqueeze if unsqueezed
            if unsqueezed and tensor.shape[1] == 1 and tensor.ndim == 2:
                tensor = tensor.squeeze(1)

            return tensor

        return inner

    @_with_tensor_format
    def normalize(self, tensor: Tensor, target: NormTarget) -> Tensor:
        """
        Apply normalization to a tensor.

        Args:
            tensor (Tensor): Feature or label tensor.
            target (NormTarget): Whether this tensor represents features or labels.

        Returns:
            Tensor: Normalized tensor (same shape).
        """
        return self._normalize(tensor, target)

    @_with_tensor_format
    def inverse_normalize(self, tensor: Tensor, target: NormTarget) -> Tensor:
        """
        Apply inverse normalization to a tensor.

        Args:
            tensor (Tensor): Feature or label tensor.
            target (NormTarget): Whether this tensor represents features or labels.

        Returns:
            Tensor: De-normalized tensor (same shape).
        """
        return self._inverse_normalize(tensor, target)

    @abstractmethod
    def _normalize(self, tensor: Tensor, target: NormTarget) -> Tensor:
        ...

    @abstractmethod
    def _inverse_normalize(self, tensor: Tensor, target: NormTarget) -> Tensor:
        ...