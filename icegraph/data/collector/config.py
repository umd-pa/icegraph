# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

__all__ = ["CollectorConfig"]


class CollectorConfig(BaseModel):
    # a block closes only after its tables reach this size, 0 gives one file with events per block
    min_bytes:  int                         = Field(default=0, ge=0)

    # files are only collected with files of the same group
    # parent: files in one directory, all: every file
    group_by:   Literal["parent", "all"]    = "parent"
