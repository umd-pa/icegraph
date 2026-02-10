# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from pydantic import BaseModel, Field

from .types import ReductionType

__all__ = ["MSELossConfig", "L1LossConfig", "CrossEntropyLossConfig", "BCEWithLogitsLossConfig", "NLLLossConfig"]


class MSELossConfig(BaseModel):
    reduction: ReductionType = "mean"


class L1LossConfig(BaseModel):
    reduction: ReductionType = "mean"


class CrossEntropyLossConfig(BaseModel):
    reduction:          ReductionType       = "mean"
    weight:             list[float] | None  = None
    ignore_index:       int                 = -100
    label_smoothing:    float               = Field(default=0.0, ge=0.0, le=1.0)


class BCEWithLogitsLossConfig(BaseModel):
    reduction:  ReductionType       = "mean"
    weight:     list[float] | None  = None
    pos_weight: list[float] | None  = None


class NLLLossConfig(BaseModel):
    reduction:      ReductionType       = "mean"
    weight:         list[float] | None  = None
    ignore_index:   int                 = -100
