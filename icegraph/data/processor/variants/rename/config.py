# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, Field, model_validator

from ...types import Columns

__all__ = ["RenameConfig"]


class RenameConfig(BaseModel):
    map_: dict[str | int, str | int] | None = Field(alias="map", default=None)

    # alternative mode
    cols:   Columns | None = None
    out:    Columns | None = None

    @model_validator(mode="after")
    def ensure_single_method(self) -> Self:
        has_mapping = self.map_ is not None
        has_pair = self.cols is not None or self.out is not None

        # must choose exactly one mode
        if not has_mapping and not has_pair:
            raise ValueError("Must provide either 'map' or 'cols'/'out'.")

        if has_mapping and has_pair:
            raise ValueError("Provide either 'map' or 'cols'/'out', not both.")

        if not has_mapping:
            if self.cols is None or self.out is None:
                raise ValueError("Both 'cols' and 'out' must be provided together.")

        return self



