# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import final, ClassVar, Generic, TypeVar
from abc import abstractmethod
from collections import deque
from functools import cached_property

import torch
from torch import Tensor

from icegraph.common.data import Split
from icegraph.common.tensors import SegmentedTensor
from icegraph.common.plugins import Plugin

from ..types import MetricValue, HeadValues

from .types import MetricContext

__all__ = ["Metric"]


C = TypeVar("C")  # plugin config type (unchanged)
S = TypeVar("S")  # accumulator state chosen freely by each metric


class Metric(Plugin[C, MetricContext], Generic[C, S]):
    """
    Base class for task-specific metrics.

    A metric is expressed as a monoid over a plugin-chosen accumulator
    state ``S``::

        initial()                 -> S            identity / empty accumulator
        update_state(s, out, tgt) -> S            fold one batch into the accumulator
        combine(a, b)             -> S            merge two accumulators (DDP / parallel)
        finalize(s)               -> HeadValues   resolve accumulator to per-head values

    Laws plugins must uphold:
      * ``combine`` is associative (up to fp tolerance)
      * ``initial()`` is its identity: ``combine(initial(), x) == x``

    so that batchwise accumulation equals an all-at-once computation and the
    DDP reduction is correct regardless of shard boundaries.

    ``finalize`` returns one entry per head, positionally. Each entry is either
    a 1-D tensor -- length chosen by the metric, and fixed for the lifetime of
    the run -- or ``None`` for a head that has no value this epoch. The arity of
    the tuple and the shape of each present slot are checked for stability
    across epochs, since the summaries below are elementwise over them.

    Summary semantics when a slot is ``None``: the EMA holds its last known
    value (a gap does not erase the estimate), while the delta is ``None``
    (rate of change is undefined with no current value).
    """
    SPAN: ClassVar[int] = 5

    _state:     dict[Split, S]
    _finalized: dict[Split, HeadValues]
    _ema:       dict[Split, HeadValues]
    _delta:     dict[Split, HeadValues]
    _window:    dict[Split, deque[HeadValues]]
    _layout:    dict[Split, tuple[torch.Size | None, ...]]

    @final
    def build(self) -> None:
        # accumulator starts at the monoid identity
        self._state     = {s: self.initial() for s in Split.all()}
        self._finalized = {}

        # hyper metrics
        self._ema    = {}
        self._delta  = {}
        self._window = {s: deque(maxlen=type(self).SPAN) for s in Split.all()}

        # per-slot reference shapes, learned on first sighting
        self._layout = {}

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()

        if not isinstance(cls.SPAN, int) or cls.SPAN < 1:
            raise TypeError(f"{cls.__name__}.SPAN must be a positive non-zero integer.")

    @cached_property
    def _ema_alpha(self) -> float:
        return 2.0 / (type(self).SPAN + 1)

    def _value(self, split: Split) -> HeadValues:
        """Finalized metric value, one entry per head."""
        value = self._finalized.get(split)

        if value is None:
            value = self.finalize(self._state[split])
            self._check(split, value)

            # detach at the boundary so nothing downstream pins the graph
            value = self._finalized[split] = tuple(
                None if v is None else v.detach() for v in value
            )

        return value

    @final
    def _check(self, split: Split, value: HeadValues) -> None:
        """Validate arity and per-slot shape against what this split has seen."""
        name = type(self).__name__

        if not isinstance(value, tuple):
            raise TypeError(f"{name}.finalize must return a tuple, got {type(value).__name__}.")

        layout = self._layout.get(split)

        if layout is None:
            layout = (None,) * len(value)
        elif len(layout) != len(value):
            raise ValueError(
                f"{name}.finalize returned {len(value)} heads for {split}, "
                f"but previously returned {len(layout)}. Head arity must be fixed."
            )

        resolved = list(layout)

        for i, v in enumerate(value):
            if v is None:
                continue

            if not isinstance(v, Tensor):
                raise TypeError(f"{name}.finalize slot {i} must be a Tensor or None, got {type(v).__name__}.")

            if v.ndim != 1:
                raise ValueError(
                    f"{name}.finalize slot {i} must be 1-D, got shape {tuple(v.shape)}. "
                    f"Use .reshape(1) for a single value."
                )

            ref = layout[i]

            if ref is None:
                resolved[i] = v.shape
            elif ref != v.shape:
                raise ValueError(
                    f"{name}.finalize slot {i} changed shape for {split}: "
                    f"{tuple(ref)} -> {tuple(v.shape)}. Slot widths must be fixed."
                )

        self._layout[split] = tuple(resolved)

    @final
    def _invalidate(self, split: Split) -> None:
        # drop the cached finalize result
        self._finalized.pop(split, None)

    @final
    def update(self, out: SegmentedTensor, target: SegmentedTensor, split: Split) -> None:
        """Fold a new batch into the accumulator."""
        self._state[split] = self.update_state(self._state[split], out, target)
        self._invalidate(split)

    @final
    def merge(self, other: Metric[C, S], split: Split) -> None:
        """Merge another metric's accumulator into this one in-place."""
        self._state[split] = self.combine(self._state[split], other._state[split])
        self._invalidate(split)

    @final
    def reset(self, split: Split) -> None:
        """Reset the accumulator (called at the start of each epoch)."""
        self._state[split] = self.initial()
        self._invalidate(split)

    @final
    def compute(self, split: Split) -> MetricValue:
        """Return the finalized value plus smoothing metadata."""
        ema = self._ema.get(split)
        if ema is None:
            raise ValueError("EMA has not been computed, did you call update_summaries?")

        delta = self._delta.get(split)
        if delta is None:
            raise ValueError("Delta has not been computed, did you call update_summaries?")

        return MetricValue(
            repr=self.repr(),
            value=self._to_cpu(self._value(split)),
            ema=self._to_cpu(ema),
            delta=self._to_cpu(delta),
            span=type(self).SPAN,
            optimum=self.optimum
        )

    @final
    def update_summaries(self, split: Split) -> None:
        value  = self._value(split)
        window = self._window[split]

        # rolling window; clone so an in-place accumulator cannot rewrite history
        window.append(tuple(None if v is None else v.clone() for v in value))

        self._ema[split]   = self._advance_ema(self._ema.get(split), value)
        self._delta[split] = self._window_delta(window, value)

    @final
    def _advance_ema(self, prev: HeadValues | None, value: HeadValues) -> HeadValues:
        """Blend elementwise per head; a ``None`` slot holds its previous estimate."""
        alpha = self._ema_alpha

        if prev is None:
            prev = (None,) * len(value)

        out: list[Tensor | None] = []

        for p, v in zip(prev, value):
            if v is None:
                out.append(p)
            elif p is None:
                out.append(v.clone())
            else:
                out.append(alpha * v + (1.0 - alpha) * p)

        return tuple(out)

    @staticmethod
    def _window_delta(window: deque[HeadValues], value: HeadValues) -> HeadValues:
        """Average per-step change across the window, per head, skipping gaps."""
        out: list[Tensor | None] = []

        for i, v in enumerate(value):
            if v is None:
                out.append(None)
                continue

            present = [(k, entry[i]) for k, entry in enumerate(window) if entry[i] is not None]

            if len(present) < 2:
                out.append(torch.zeros_like(v))
            else:
                (first_at, first), (last_at, last) = present[0], present[-1]

                # assertions
                assert last is not None
                assert first is not None

                out.append((last - first) / (last_at - first_at))

        return tuple(out)

    @staticmethod
    def _to_cpu(value: HeadValues) -> HeadValues:
        return tuple(None if v is None else v.cpu() for v in value)

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
    def finalize(self, state: S) -> HeadValues:
        """Resolve the accumulator to per-head values, one tuple entry per head."""
        ...