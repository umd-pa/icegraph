# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

from torch import Tensor

from icegraph.common.data import GraphBatch, ProcessedGraphBatch

if TYPE_CHECKING:
    from icegraph.inference import Inference


Loss: TypeAlias = float | Tensor
EpochLoss: TypeAlias = float


# BASE CONTEXTS
@dataclass(frozen=True, slots=True)
class Context:
    inference: Inference


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
class TeardownContext(Context): ...


# BATCH HOOK CONTEXTS
@dataclass(frozen=True, slots=True)
class BatchBeginContext(GraphBatchContext): ...


@dataclass(frozen=True, slots=True)
class BatchTransferContext(GraphBatchContext): ...


@dataclass(frozen=True, slots=True)
class BatchEndContext(ProcessedGraphBatchContext):
    loss: Loss
