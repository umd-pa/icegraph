# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from pydantic import BaseModel

from ...types import Columns

__all__ = ["CompressorConfig"]


class CompressorConfig(BaseModel):
    to:     str
    by:     Columns
    cols:   Columns
    out:    str | int
    dtype:  str | None = None
    record_names: bool = True
    record_offset: bool = True

    # record the packed dtype for every column instead of the one it came in with,
    # for a pack whose stored type is the one that matters
    override_dtypes: bool = False
