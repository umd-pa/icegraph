# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

from torch import Tensor

from icegraph.common.data import GraphBatch, ProcessedGraphBatch

if TYPE_CHECKING:
    from icegraph.trainer import Trainer


Loss: TypeAlias = float | Tensor
EpochLoss: TypeAlias = float


# BASE CONTEXTS
@dataclass(frozen=True, slots=True)
class Context:
    trainer: Trainer


@dataclass(frozen=True, slots=True)
class GraphBatchContext(Context):
    batch: GraphBatch


@dataclass(frozen=True, slots=True)
class ProcessedGraphBatchContext(Context):
    batch: ProcessedGraphBatch


# TRAINER-ONLY HOOK CONTEXTS
@dataclass(frozen=True, slots=True)
class InitContext(Context): ...


@dataclass(frozen=True, slots=True)
class ExecuteContext(Context): ...


@dataclass(frozen=True, slots=True)
class EpochBeginContext(Context): ...


@dataclass(frozen=True, slots=True)
class EpochEndContext(Context): ...


@dataclass(frozen=True, slots=True)
class TrainBeginContext(Context): ...


@dataclass(frozen=True, slots=True)
class ValidationBeginContext(Context): ...


@dataclass(frozen=True, slots=True)
class TestBeginContext(Context): ...


@dataclass(frozen=True, slots=True)
class TeardownContext(Context): ...


# BATCH HOOK CONTEXTS
@dataclass(frozen=True, slots=True)
class BatchBeginContext(GraphBatchContext): ...


@dataclass(frozen=True, slots=True)
class BatchEndContext(ProcessedGraphBatchContext):
    loss: Loss


# END-OF-PHASE CONTEXTS
@dataclass(frozen=True, slots=True)
class TrainEndContext(Context):
    loss: EpochLoss


@dataclass(frozen=True, slots=True)
class ValidationEndContext(Context):
    loss: EpochLoss


@dataclass(frozen=True, slots=True)
class TestEndContext(Context):
    loss: EpochLoss
