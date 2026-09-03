# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import TYPE_CHECKING, Generic, TypeVar
from abc import ABC, abstractmethod
from pathlib import Path
from functools import cached_property
from collections.abc import Mapping

import torch
from torch import Tensor

from icegraph.common.transforms import TransformSpace
from icegraph.trainer.callbacks import TrainerCallback
from icegraph.common.data import Split, DataRole

from ..accumulator import Accumulator

if TYPE_CHECKING:
    from icegraph.trainer import Trainer
    from icegraph.trainer.callbacks import context

__all__ = ["Reducer"]

import logging
logger = logging.getLogger(__name__)


T = TypeVar("T")
A = TypeVar("A", bound="Accumulator")


class Reducer(TrainerCallback, ABC, Generic[T, A]):
    """
    Base class for online data reduction during testing/validation splits.

    Reducers accumulate batch-level data and emit reduced artifacts
    (e.g. histograms).
    """

    _ctx:               context.InitContext

    def __init__(self, **kwargs) -> None:
        super().__init__()

        # store kwargs
        self._kwargs = kwargs

        # dict of accumulators
        self._accumulators: dict[str, dict[int, A]] = {}

    def on_init(self, ctx: context.InitContext) -> None:
        # cache ctx
        self._ctx = ctx

        # break out if not rank 0
        if not ctx.engine.state.is_main_process():
            return

    @cached_property
    def _target_labels(self) -> list[str]:
        return self._ctx.engine.decode.get_columns(DataRole.TARGETS)

    def on_batch_end(self, ctx: context.BatchEndContext) -> None:
        trainer = ctx.engine

        # break out if not rank 0 and in eval
        if not trainer.state.is_main_process() or trainer.split not in Split.eval():
            return

        for out, target, label in zip(
            ctx.batch.out,
            ctx.batch.targets,
            ctx.batch.out.names,
            strict=True
        ):
            # reduce data using subclass logic
            reduced = self._reduce(out, target, ctx)

            if isinstance(reduced, tuple):
                data, acc_idx = reduced

                # ensure acc_idx is torch.long
                acc_idx = acc_idx.long()
            else:
                # if only one accumulator, set each sample to map to it
                data = reduced
                acc_idx = torch.zeros(data.size(0), dtype=torch.long, device=data.device)

            if data.ndim != 2:
                raise ValueError(
                    f"{type(self).__name__}._reduce must return data with shape [B, D]. "
                    f"For 1D data, use shape [B, 1], not [B]. "
                    f"Got shape {tuple(data.shape)}."
                )

            if acc_idx.ndim != 1:
                raise ValueError(
                    f"{type(self).__name__}._reduce must return acc_idx with shape [B]. "
                    f"Got shape {tuple(acc_idx.shape)}."
                )

            if data.size(0) != acc_idx.size(0):
                raise ValueError(
                    f"{type(self).__name__}._reduce returned mismatched batch sizes: "
                    f"data.shape={tuple(data.shape)}, acc_idx.shape={tuple(acc_idx.shape)}."
                )

            for i in torch.unique(acc_idx):
                # encode data to accumulator format
                encoded = self._encode(data[acc_idx == i], label)

                # update the central accumulators
                (self._accumulators
                    .setdefault(label, {})
                    .setdefault(int(i.item()), self._build_accumulator())
                    .update(encoded)
                )

    def finalize(self, trainer: Trainer) -> None:
        # iterate over artifacts and dispatch
        for label, acc_bundle in self._accumulators.items():
            # perform any postprocessing
            processed_acc_bundle = self._postprocess_accumulator(acc_bundle, label)

            # get any empty accumulators
            empty = [a.is_empty() for a in processed_acc_bundle.values()]

            # if all are empty, warn and skip
            if all(empty):
                logger.warning(
                    "%s has no data for label %r; all accumulators are empty. Skipping dispatch.",
                    type(self).__name__,
                    label
                )
                continue

            # if some are empty, warn but dispatch
            if any(empty):
                empty_indices = [
                    idx for idx, acc in processed_acc_bundle.items()
                    if acc.is_empty()
                ]

                logger.warning(
                    "%s has empty accumulator(s) for label %r: %s. "
                    "Dispatching remaining non-empty accumulators.",
                    type(self).__name__,
                    label,
                    empty_indices
                )

            # build all artifacts
            artifacts: dict[int | str, T] = {}
            space: tuple[TransformSpace, ...] | None = None
            for i, a in processed_acc_bundle.items():
                if a.is_empty():
                    # skip over empty accumulators
                    continue

                artifacts[i], _space = self._build_artifact(a, label)

                if space is None:
                    space = _space

                elif _space != space:
                    raise ValueError(f"Space for artifact {i} ({_space}) is not equal to expected space ({space}).")

            assert space is not None
            self._dispatch(trainer, artifacts, space, label)

        # reset for the next split
        self.reset()

    def reset(self) -> None:
        # reset all accumulators
        for acc_bundle in self._accumulators.values():
            for a in acc_bundle.values():
                a.reset()

    # link callback hooks to methods (only call on validation and test splits)
    def on_validation_end(self, ctx: context.ValidationEndContext) -> None:
        # break out if not rank 0
        if not ctx.engine.state.is_main_process():
            return

        self.finalize(ctx.engine)

    def on_test_end(self, ctx: context.TestEndContext) -> None:
        # break out if not rank 0
        if not ctx.engine.state.is_main_process():
            return
        
        self.finalize(ctx.engine)

    ### Abstract methods for subclassing ###

    def _postprocess_accumulator(self, data: Mapping[int, A], label: str) -> Mapping[int, A] | Mapping[str, A]:
        return data

    @abstractmethod
    def _build_accumulator(self) -> A:
        ...

    @abstractmethod
    def _build_artifact(self, accumulator: A, label: str) -> tuple[T, tuple[TransformSpace, ...]]:
        ...

    @abstractmethod
    def _encode(self, data: Tensor, label: str) -> Tensor:
        ...

    @abstractmethod
    def _reduce(self, out: Tensor, target: Tensor, ctx: context.BatchEndContext) -> Tensor | tuple[Tensor, Tensor]:
        ...

    @abstractmethod
    def _dispatch(self, trainer: Trainer, data: dict[int | str, T], space: tuple[TransformSpace, ...], label: str) -> None:
        ...
