# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from icegraph.engine.callbacks import Context, InitContext  # import without usage is intentional

if TYPE_CHECKING:
    from ..inference import BatchInference


@dataclass(frozen=True, slots=True)
class InferenceContext(Context["BatchInference"]): ...


@dataclass(frozen=True, slots=True)
class ExecuteContext(InferenceContext): ...


@dataclass(frozen=True, slots=True)
class TeardownContext(InferenceContext): ...
