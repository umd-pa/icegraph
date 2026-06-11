# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import final, ClassVar, Generic, TypeVar
from abc import abstractmethod
from collections import deque
from functools import cached_property

import torch
from torch import Tensor

from icegraph.common.tensors import SegmentedTensor
from icegraph.common.plugins import Plugin

from ..types import ComputedMetric

from .types import MetricContext

__all__ = ["Metric"]


C = TypeVar("C")  # plugin config type (unchanged)
S = TypeVar("S")  # accumulator state — chosen freely by each metric


class Metric(Plugin[C, MetricContext], Generic[C, S]):
    """
    Base class for task-specific metrics.

    A metric is expressed as a monoid over a plugin-chosen accumulator
    state ``S``::

        initial()                 -> S       identity / empty accumulator
        update_state(s, out, tgt) -> S       fold one batch into the accumulator
        combine(a, b)             -> S       merge two accumulators (DDP / parallel)
        finalize(s)               -> Tensor  resolve accumulator to a per-head value

    Laws plugins must uphold:
      * ``combine`` is associative (up to fp tolerance)
      * ``initial()`` is its identity: ``combine(initial(), x) == x``

    so that batchwise accumulation equals an all-at-once computation and the
    DDP reduction is correct regardless of shard boundaries.
    """
    SPAN: ClassVar[int] = 5

    _state:     S
    _ema:       Tensor | None
    _delta:     Tensor | None

    @final
    def build(self) -> None:
        # accumulator starts at the monoid identity
        self._state = self.initial()

        # hyper metrics
        self._ema   = None
        self._delta = None

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()

        if not isinstance(cls.SPAN, int) or cls.SPAN < 1:
            raise TypeError(f"{cls.__name__}.SPAN must be a positive non-zero integer.")

    @cached_property
    def _ema_alpha(self) -> float:
        return 2.0 / (type(self).SPAN + 1)

    @cached_property
    def _window(self) -> deque[Tensor]:
        # window of recent values for delta (holds [L] tensors)
        return deque(maxlen=type(self).SPAN)

    @cached_property
    def computed(self) -> Tensor:
        """Finalized metric value."""
        return self.finalize(self._state)

    @final
    def _invalidate(self) -> None:
        # drop the cached finalize result
        self.__dict__.pop("computed", None)

    @final
    def update(self, out: SegmentedTensor, target: SegmentedTensor) -> None:
        """Fold a new batch into the accumulator."""
        self._state = self.update_state(self._state, out, target)
        self._invalidate()

    @final
    def merge(self, other: Metric[C, S]) -> None:
        """Merge another metric's accumulator into this one in-place."""
        self._state = self.combine(self._state, other._state)
        self._invalidate()

    @final
    def reset(self) -> None:
        """Reset the accumulator (called at the start of each epoch)."""
        self._state = self.initial()
        self._invalidate()

    @final
    def compute(self) -> ComputedMetric:
        """Return the finalized value plus smoothing metadata."""
        return ComputedMetric(
            repr=self.repr(),
            value=self.computed.cpu(),
            ema=None if self._ema is None else self._ema.cpu(),
            delta=None if self._delta is None else self._delta.cpu(),
            span=type(self).SPAN,
            optimum=self.optimum
        )

    @final
    def update_summaries(self) -> None:
        value = self.computed

        # rolling window
        self._window.append(value.detach().clone())

        # ema elementwise over heads
        if self._ema is None:
            self._ema = value.detach().clone()
        else:
            self._ema = self._ema_alpha * value + (1.0 - self._ema_alpha) * self._ema

        # average per-step delta over the window, elementwise over heads
        n = len(self._window)
        if n < 2:
            self._delta = torch.zeros_like(value)
        else:
            self._delta = (self._window[-1] - self._window[0]) / (n - 1)

    ### PLUGIN

    @property
    @abstractmethod
    def optimum(self) -> float:
        """Float representation of the optimum value for this metric."""
        ...

    @abstractmethod
    def repr(self) -> str:
        """String representation of the metric being computed."""
        ...

    @abstractmethod
    def initial(self) -> S:
        """Create an empty accumulator."""
        ...

    @abstractmethod
    def update_state(self, state: S, out: SegmentedTensor, target: SegmentedTensor) -> S:
        """Fold one batch into ``state`` and return the updated accumulator."""
        ...

    @abstractmethod
    def combine(self, a: S, b: S) -> S:
        """Merge two accumulators. Associative, with ``initial()`` as identity."""
        ...

    @abstractmethod
    def finalize(self, state: S) -> Tensor:
        """Resolve the accumulator to a per-head value tensor (shape [L])."""
        ...