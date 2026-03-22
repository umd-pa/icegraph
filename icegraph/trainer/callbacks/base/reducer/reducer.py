# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterator
from abc import ABC, abstractmethod
from pathlib import Path

import torch
from torch import Tensor

from icegraph.statistics import StatisticService
from icegraph.trainer.callbacks.base.accumulator import AccumulatorStore
from icegraph.trainer.callbacks import Callback
from icegraph.trainer.shared import GroupTransform, TransformSpec, GroupTransformSpec
from icegraph.types.transforms import TransformSpace
from icegraph.types.data import ModelInputRole, Split

if TYPE_CHECKING:
    from icegraph.trainer import Trainer
    from icegraph.trainer.callbacks import context

__all__ = ["Reducer"]


class Reducer(Callback, ABC):
    """
    Base class for online data reduction during testing/validation splits.

    Reducers accumulate batch-level data and emit reduced artifacts
    (e.g. histograms) that are later consumed by renderers.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__()

        # cache user plot scale choice
        _axis_scale: tuple[str, str] = kwargs.get("scale") or ("linear", "linear")
        self._axis_scale: tuple[TransformSpace, TransformSpace] = (
            TransformSpace(_axis_scale[0]),
            TransformSpace(_axis_scale[1]),
        )

        # group transform
        self._transform = GroupTransform()

        # device
        self.device: torch.device | None = None

        # main accumulator
        self._accumulator: AccumulatorStore | None = None

        # cache label stats
        self._stats: StatisticService | None = None

        # save dir
        self._save_dir: Path | None = None

        # cache training labels
        self._target_labels:    list[str] | None = None
        self._auxiliary_labels: list[str] | None = None

    def on_init(self, ctx: context.InitContext) -> None:
        trainer = ctx.trainer

        # cache the trainer device (will not change over a run)
        self.device = trainer.state.device

        # load aggregate label stats
        self._stats = trainer.data.stats(Split.TRAIN, ModelInputRole.LABELS)

        # grab target and auxiliary labels
        self._target_labels = trainer.data.columns(ModelInputRole.LABELS)
        self._auxiliary_labels = trainer.data.columns(ModelInputRole.LABELS, aux=True)

        # grab trainer output directory and build plot dir
        self._save_dir = trainer.outdir / "plots"
        self._save_dir.mkdir(parents=True, exist_ok=True)

        # configure the group transformer
        self._configure_transformer()

        self._post_init(trainer)

    def _configure_transformer(self) -> None:
        # build specs
        specs: list[TransformSpec] = []
        for space in self._axis_scale:
            specs.append(TransformSpec(space, base=10))

        # configure group transform
        self._transform.configure_from_spec(GroupTransformSpec(specs))

        # move transform to device
        self._transform.to(self.device)

    def on_batch_end(self, ctx: context.BatchEndContext) -> None:
        trainer = ctx.trainer
        # skip if not in eval
        if trainer.split == Split.TRAIN:
            return

        # reduce data using client logic and accumulate
        reduced_data = self._reduce(ctx)

        # run any internal post reduction
        post_reduced_data = self._post_reduce(reduced_data)

        # send to accumulator
        self._accumulate(post_reduced_data)

    def _accumulate(self, data: Tensor) -> None:
        # build the accumulator lazily on first call of each split
        if self._accumulator is None:
            self._accumulator = self._init_accumulator(self.device)

        # encode data to accumulator format
        encoded_data = self._encode(data)

        # add to the central accumulator
        self._accumulator += encoded_data

    def finalize(self, trainer: Trainer) -> None:
        # dont do anything if there is no accumulator
        if self._accumulator is None:
            return

        # iterate over artifacts (user decides how to build artifacts) and dispatch
        for item in self._emit_artifacts(trainer, self._accumulator):
            self._dispatch(trainer, item)

        # reset for the next split
        self.reset()

    def reset(self) -> None:
        # wipe the tensor store
        self._accumulator = None

    # link callback hooks to methods (only call on validation and test splits)
    def on_validation_end(self, ctx: context.ValidationEndContext) -> None:
        self.finalize(ctx.trainer)

    def on_test_end(self, ctx: context.TestEndContext) -> None:
        self.finalize(ctx.trainer)

    ### Abstract methods for subclassing ###

    def _post_init(self, trainer: Trainer) -> None:
        # default: dont do anything
        pass

    def _post_reduce(self, data: Tensor) -> Tensor:
        # default: dont do anything
        return data

    @abstractmethod
    def _encode(self, indices: Tensor) -> dict[str, Tensor]:
        ...

    @abstractmethod
    def _emit_artifacts(self, trainer: Trainer, accumulator: AccumulatorStore) -> Iterator[Any]:
        ...

    @abstractmethod
    def _init_accumulator(self, device: torch.device) -> AccumulatorStore:
        ...

    @abstractmethod
    def _reduce(self, ctx: context.BatchEndContext) -> Tensor:
        ...

    @abstractmethod
    def _dispatch(self, trainer: Trainer, data: Any) -> None:
        ...
