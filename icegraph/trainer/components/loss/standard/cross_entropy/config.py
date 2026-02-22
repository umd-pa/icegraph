# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from pydantic import BaseModel, Field

from icegraph.trainer.components.loss.types import ReductionType

__all__ = ["Config"]


class Config(BaseModel):
    reduction:          ReductionType       = "mean"
    weight:             list[float] | None  = None
    ignore_index:       int                 = -100
    label_smoothing:    float               = Field(default=0.0, ge=0.0, le=1.0)
