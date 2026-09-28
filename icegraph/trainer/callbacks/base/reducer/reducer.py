# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import TYPE_CHECKING, Generic, TypeVar, final
from abc import ABC, abstractmethod
from collections.abc import Mapping

import torch
from torch import Tensor

from icegraph.trainer.callbacks import TrainerCallback
from icegraph.common.data import Split

if TYPE_CHECKING:
    from icegraph.trainer import Trainer
    from icegraph.trainer.callbacks import context

__all__ = ["Reducer"]

import logging
logger = logging.getLogger(__name__)


S = TypeVar("S")  # accumulator state chosen freely by each reducer
T = TypeVar("T")  # artifact emitted per named series


class Reducer(TrainerCallback, ABC, Generic[S, T]):
    """
    Base class for online data reduction during validation/test splits.

    A reducer is expressed as a monoid over a reducer-chosen accumulator
    state ``S``, kept per ``(label, group)``:

        project(out, target)        -> rows | (rows, groups)   map one head's batch to rows [B, D] and group keys [B]
        initial()                   -> S                       identity / empty accumulator
        update_state(s, rows, lbl)  -> S                       fold one group's rows into the accumulator
        combine(a, b)               -> S                       merge two accumulators (DDP / parallel)
        finalize(states, lbl)       -> {name: T}               resolve a label's group accumulators to named artifacts
        emit(trainer, artifacts, lbl)                          publish the artifacts (plot, export, ...)

    Laws reducers must uphold:
      * ``combine`` is associative (up to fp tolerance)
      * ``initial()`` is its identity: ``combine(initial(), x) == x``

    Groups only exist once a row has been routed to them, so ``finalize`` never
    sees a group that received no data. A reducer whose ``project`` returns bare
    rows places everything in group ``0``.
    """

    _ctx:       context.InitContext
    _states:    dict[str, dict[int, S]]

    def __init__(self, **kwargs) -> None:
        super().__init__()

        # store kwargs
        self._kwargs = kwargs

        # accumulators, keyed by label then group
        self._states = {}

    def on_init(self, ctx: context.InitContext) -> None:
        # cache ctx
        self._ctx = ctx

    @final
    def update(self, out: Tensor, target: Tensor, label: str) -> None:
        """Project one head's batch and fold it into the per-group accumulators."""
        projected = self.project(out, target)

        cls = type(self).__name__

        rows: Tensor
        groups: Tensor
        if isinstance(projected, tuple):
            # groups explicitly provided
            if any(~torch.is_tensor(t) for t in projected):
                raise TypeError(
                    f"{cls}.project must return a Tensor or a tuple of Tensors, "
                    f"got {tuple(type(t).__name__ for t in projected)}."
                )

            rows, groups = projected
            groups = groups.long()
        else:
            # everything maps to a single group
            if not torch.is_tensor(projected):
                raise TypeError(
                    f"{cls}.project must return a Tensor or a tuple of Tensors, "
                    f"got {type(projected).__name__!r}."
                )
            rows = projected
            groups = torch.zeros(rows.size(0), dtype=torch.long, device=rows.device)

        if rows.ndim != 2:
            raise ValueError(
                f"{cls}.project must return rows with shape [B, D] (use [B, 1] for 1D data), "
                f"got shape {tuple(rows.shape)}."
            )

        if groups.shape != rows.shape[:1]:
            raise ValueError(
                f"{cls}.project must return groups with shape [B] matching rows shape [B, D], "
                f"got rows.shape={tuple(rows.shape)}, groups.shape={tuple(groups.shape)}."
            )

        if rows.size(0) == 0:
            return

        # partition rows by group in one pass
        sorted_groups, order = torch.sort(groups, stable=True)
        keys, counts = torch.unique_consecutive(sorted_groups, return_counts=True)
        chunks = rows[order].split(counts.tolist())

        states = self._states.setdefault(label, {})

        for key, chunk in zip(keys.tolist(), chunks, strict=True):
            states[key] = self.update_state(states.get(key, self.initial()), chunk, label)

    @final
    def merge(self, other: Reducer[S, T]) -> None:
        """Merge another reducer's accumulators into this one in-place."""
        for label, theirs in other._states.items():
            ours = self._states.setdefault(label, {})

            for key, state in theirs.items():
                ours[key] = self.combine(ours.get(key, self.initial()), state)

    @final
    def reset(self) -> None:
        """Drop all accumulators."""
        self._states.clear()

    @final
    def flush(self, trainer: Trainer) -> None:
        """Finalize and emit every label, then reset for the next split."""
        for label, states in self._states.items():
            artifacts = self.finalize(states, label)

            # finalize may legitimately resolve to nothing, for example ROC needs both the
            # positive and negative group of a class before it can draw a curve
            if not artifacts:
                logger.warning(
                    "%s has nothing to emit for label %r after finalize. Skipping.",
                    type(self).__name__,
                    label
                )
                continue

            self.emit(trainer, artifacts, label)

        self.reset()

    ### Callback hooks (evaluation splits, main process only) ###

    def on_batch_end(self, ctx: context.BatchEndContext) -> None:
        trainer = ctx.engine

        if not trainer.state.is_main_process() or trainer.split not in Split.eval():
            return

        for out, target, label in zip(ctx.batch.out, ctx.batch.targets, ctx.batch.out.names, strict=True):
            self.update(out, target, label)

    def on_validation_end(self, ctx: context.ValidationEndContext) -> None:
        if ctx.engine.state.is_main_process():
            self.flush(ctx.engine)

    def on_test_end(self, ctx: context.TestEndContext) -> None:
        if ctx.engine.state.is_main_process():
            self.flush(ctx.engine)

    ### Abstract methods for subclassing ###

    @abstractmethod
    def project(self, out: Tensor, target: Tensor) -> Tensor | tuple[Tensor, Tensor]:
        """Map one head's batch to rows ``[B, D]``, optionally with group keys ``[B]``."""
        ...

    @abstractmethod
    def initial(self) -> S:
        """Create an empty accumulator."""
        ...

    @abstractmethod
    def update_state(self, state: S, rows: Tensor, label: str) -> S:
        """Fold one group's rows into ``state`` and return the updated accumulator."""
        ...

    @abstractmethod
    def combine(self, a: S, b: S) -> S:
        """Merge two accumulators. Associative, with ``initial()`` as identity."""
        ...

    @abstractmethod
    def finalize(self, states: Mapping[int, S], label: str) -> dict[str, T]:
        """Resolve a label's group accumulators to named artifacts; may be empty."""
        ...

    @abstractmethod
    def emit(self, trainer: Trainer, artifacts: dict[str, T], label: str) -> None:
        """Publish the finalized artifacts for ``label``."""
        ...
