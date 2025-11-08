# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Optional, final, Tuple, TYPE_CHECKING
from abc import ABC, abstractmethod

from torch import nn
import torch

from icegraph.types import ComputedMetrics

if TYPE_CHECKING:
    from icegraph.trainer import Trainer
else:
    Trainer = None


class TaskStrategy(ABC):
    """
    Base class for task-specific training strategies.

    A TaskStrategy defines how a model computes loss, adapts targets,
    determines input/output channels, and constructs metrics for a given task.
    """
    task: str

    def __init__(self, **kwargs) -> None:
        """
        Initialize a task strategy.

        Args:
            **kwargs: Arbitrary keyword arguments passed to the strategy.
        """
        self._enforced_reduction = "sum"
        self.kwargs = kwargs

    def __init_subclass__(cls, **kwargs) -> None:
        """
        Enforce required attributes on subclasses.

        Raises:
            NotImplementedError: If subclass does not define the `task` attribute.
        """
        super().__init_subclass__(**kwargs)

        for attr in ["task"]:
            if getattr(cls, attr, None) is None:
                raise NotImplementedError(f"Subclasses of 'TaskStrategy' must implement the '{attr}' class attribute.")

    @abstractmethod
    def loss_function(self) -> nn.Module:
        """
        Return the loss function for this task.

        Returns:
            nn.Module: Torch loss module.
        """
        ...

    @abstractmethod
    def adapt_targets(self, batch: torch.Batch, out: torch.Tensor) -> torch.Tensor:
        """
        Adapt raw batch targets to align with model outputs.

        Args:
            batch (torch.Batch): PyTorch Geometric batch.
            out (torch.Tensor): Model predictions.

        Returns:
            torch.Tensor: Adapted targets shaped to match outputs.
        """
        ...

    @abstractmethod
    def make_metrics(self) -> Metrics:
        """
        Construct metrics object for this task.

        Returns:
            Metrics: Task-specific metrics tracker.
        """
        ...

    @abstractmethod
    def out_channels(self, trainer: Trainer) -> int:
        """
        Return the number of output channels for the model.

        Args:
            trainer (Trainer): Trainer context.

        Returns:
            int: Number of output channels.
        """
        ...

    @abstractmethod
    def in_channels(self, trainer: Trainer) -> int:
        """
        Return the number of input channels for the model.

        Args:
            trainer (Trainer): Trainer context.

        Returns:
            int: Number of input channels.
        """
        ...

    @abstractmethod
    def filter_eval(
        self,
        out: torch.Tensor,
        target: torch.Tensor
    ) -> Tuple:
        """
        Filter model outputs and targets for evaluation.

        Args:
            out (torch.Tensor): Model predictions.
            target (torch.Tensor): Ground-truth targets.

        Returns:
            Tuple: Filtered `(out, target)` pair for evaluation.
        """
        ...


class Metrics(ABC):
    """
    Base class for task-specific metrics tracking.

    Metrics objects accumulate results across training steps and
    compute final aggregated values when requested.
    """
    _compute_cache: Optional[ComputedMetrics] = None

    @final
    def update(
        self,
        out: torch.Tensor,
        target: torch.Tensor,
        loss: torch.Tensor, *,
        mask: Optional[torch.Tensor] = None
    ) -> None:
        """
        Update metrics with a new batch.

        Args:
            out (torch.Tensor): Model predictions.
            target (torch.Tensor): Ground-truth targets.
            loss (torch.Tensor): Loss value for the batch.
            mask (torch.Tensor, optional): Optional mask to ignore certain samples.
        """
        self._update(out, target, loss, mask=mask)
        self._compute_cache = None

    @abstractmethod
    def _update(
        self,
        out: torch.Tensor,
        target: torch.Tensor,
        loss: torch.Tensor, *,
        mask: Optional[torch.Tensor] = None
    ) -> None:
        """
        Implement the task-specific update logic.

        Args:
            out (torch.Tensor): Model predictions.
            target (torch.Tensor): Ground-truth targets.
            loss (torch.Tensor): Loss value for the batch.
            mask (torch.Tensor, optional): Optional mask to ignore certain samples.
        """
        ...

    @final
    def compute(self) -> ComputedMetrics:
        """
        Compute and return aggregated metrics.

        Returns:
            ComputedMetrics: Aggregated metrics for the current state.
        """
        if self._compute_cache is None:
            self._compute_cache = self._compute()
        return self._compute_cache

    @abstractmethod
    def _compute(self) -> ComputedMetrics:
        """
        Implement the task-specific computation logic.

        Returns:
            ComputedMetrics: Aggregated metrics.
        """
        ...

    @abstractmethod
    def merge_(self, other: Metrics) -> None:
        """
        Merge another Metrics instance into this one in-place.

        Args:
            other (Metrics): Metrics object to merge.
        """
        ...
