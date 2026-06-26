# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from torch import Tensor

from icegraph.engine.callbacks import Context, InitContext  # import without usage is intentional
from icegraph.common.data import GraphBatch, ProcessedGraphBatch

if TYPE_CHECKING:
    from ..trainer import Trainer


@dataclass(frozen=True, slots=True)
class TrainerContext(Context["Trainer"]): ...


@dataclass(frozen=True, slots=True)
class ExecuteContext(TrainerContext): ...


@dataclass(frozen=True, slots=True)
class EpochBeginContext(TrainerContext): ...


@dataclass(frozen=True, slots=True)
class EpochEndContext(TrainerContext): ...


@dataclass(frozen=True, slots=True)
class TrainBeginContext(TrainerContext): ...


@dataclass(frozen=True, slots=True)
class ValidationBeginContext(TrainerContext): ...


@dataclass(frozen=True, slots=True)
class TestBeginContext(TrainerContext): ...


@dataclass(frozen=True, slots=True)
class TeardownContext(TrainerContext): ...


# BATCH HOOK CONTEXTS
@dataclass(frozen=True, slots=True)
class BatchBeginContext(TrainerContext):
    batch: GraphBatch


@dataclass(frozen=True, slots=True)
class BatchEndContext(TrainerContext):
    batch: ProcessedGraphBatch
    loss: Tensor


# END-OF-PHASE CONTEXTS
@dataclass(frozen=True, slots=True)
class TrainEndContext(TrainerContext):
    loss: Tensor


@dataclass(frozen=True, slots=True)
class ValidationEndContext(TrainerContext):
    loss: Tensor


@dataclass(frozen=True, slots=True)
class TestEndContext(TrainerContext):
    loss: Tensor
