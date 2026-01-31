# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Optional, final, Dict, Any
from abc import ABC, abstractmethod
from collections import deque

from torch import Tensor

from .types import ComputedMetric

__all__ = ["Metric"]


class Metric(ABC):
    """
    Base class for task-specific metrics tracking.

    Metric objects accumulate results across training steps and
    compute final aggregated values when requested.
    """
    # name for the metric (also display name on logs/console/etc)
    name: str

    # list of compatible tasks
    compatible: list[str]

    SPAN: int = 5

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

        if getattr(self, "name", None) is None:
            raise NotImplementedError(f"Subclasses of {self.__class__.__name__} must implement 'name' attribute.")

        # cache for raw data and computed metrics
        self.cache:             Dict[str, Optional[Any]]    = {}
        self._compute_cache:    Optional[float]             = None

        # hyper metrics
        self._ema:      Optional[float] = None
        self._delta:    Optional[float] = None

        # ema alpha
        self._ema_alpha: float = 2 / (self.span + 1)

        # window for delta
        self._window: deque[float] = deque(maxlen=self.span)

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        if not isinstance(cls.SPAN, int) or cls.SPAN < 1:
            raise TypeError(f"{cls.__name__}.SPAN must be a positive non-zero integer.")

    @property
    def span(self) -> int:
        return type(self).SPAN

    @final
    def update(self, out: Tensor, target: Tensor) -> None:
        """
        Update metrics with a new batch.

        Args:
            out (Tensor): Model predictions.
            target (Tensor): Ground-truth targets.
        """
        self._update(out, target)

        # wipe the compute cache as it is now invalid
        self._compute_cache = None

    @final
    def compute(self) -> ComputedMetric:
        """
        Compute and return aggregated metrics. Cached until `update()` is called again.

        Returns:
            ComputedMetric: Aggregated metrics for the current state.
        """
        if self._compute_cache is None:
            self._compute_cache = self._compute()

        computed = ComputedMetric(
            name=self.name, value=self._compute_cache, ema=self._ema, delta=self._delta, span=self.span
        )
        return computed

    @final
    def reset(self) -> None:
        """Reset metrics (called at beginning of each epoch)"""
        for key, value in self.cache.items():
            self.cache[key] = None

        self._compute_cache = None

    @final
    def update_summaries(self) -> None:
        # compute updated value
        computed = self.compute()
        value = computed.value

        # update rolling window
        self._window.append(value)

        # update ema
        if self._ema is None:
            self._ema = value
        else:
            self._ema = self._ema_alpha * value + (1 - self._ema_alpha) * self._ema

        # update average delta over window
        n = len(self._window)
        if n < 2:
            self._delta = 0.0
        else:
            self._delta = (self._window[-1] - self._window[0]) / (n - 1)

    @abstractmethod
    def _update(self, out: Tensor, target: Tensor) -> None:
        """
        Implement the metric-specific update logic.

        Args:
            out (Tensor): Model predictions.
            target (Tensor): Ground-truth targets.
        """
        ...

    @abstractmethod
    def _compute(self) -> float:
        """
        Implement the task-specific computation logic.

        Returns:
            ComputedMetric: Aggregated metrics.
        """
        ...

    @abstractmethod
    def merge(self, other: Metric) -> None:
        """
        Merge another Metric instance into this one in-place.

        Args:
            other (Metric): Metric object to merge.
        """
        ...
