# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Self

import numpy as np
from pydantic import BaseModel, Field, model_validator

__all__ = ["SplitMapConfig"]


class SplitMapConfig(BaseModel):
    seed:       int
    range_:     int                 = Field(alias="range")
    weights:    list[int | float]

    @model_validator(mode="after")
    def range_weight(self) -> Self:
        if self.range_ > 255:
            raise ValueError("range must be a number from 0 to 255")

        if len(self.weights) != self.range_:
            raise ValueError("len(weights) must equal range")

        if any(w < 0 for w in self.weights):
            raise ValueError("weights must be non-negative")

        s = sum(self.weights)

        if not np.isfinite(s) or s <= 0.0:
            raise ValueError("weights must sum to a positive finite value")

        # normalize
        self.weights = [w / s for w in self.weights]

        return self
