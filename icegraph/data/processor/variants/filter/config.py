# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, model_validator

__all__ = ["FilterConfig"]


class FilterConfig(BaseModel):
    col:        str
    condition:  Literal["gt", "lt", "ge", "le", "eq", "ne"]
    value:      int | float | str | bool

    @model_validator(mode="after")
    def validate_condition(self) -> Self:
        if isinstance(self.value, (str)) and self.condition not in {"eq", "ne"}:
            raise ValueError("Filter only supports conditions 'eq' and 'ne' for a boolean or string value.")

        return self
