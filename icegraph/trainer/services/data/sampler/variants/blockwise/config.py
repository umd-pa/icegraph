# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, model_validator

__all__ = ["Config"]


class Config(BaseModel):
    block_size: int

    @model_validator(mode="after")
    def validate_block_size(self) -> Self:
        if self.block_size <= 0:
            raise ValueError("block_size must be > 0")

        return self
