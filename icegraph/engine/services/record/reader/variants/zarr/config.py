# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, model_validator

__all__ = ["Config"]


class Config(BaseModel):
    dense_read_fraction: float = 0.5

    @model_validator(mode="after")
    def validate_dense_read_fraction(self) -> Self:

        if not (0 <= self.dense_read_fraction <= 1):
            raise ValueError("Value 'dense_read_fraction' must be a float between 0 and 1.")

        return self