# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from pydantic import BaseModel

from icegraph.trainer.components.loss.types import ReductionType

__all__ = ["Config"]


class Config(BaseModel):
    reduction: ReductionType = "mean"
