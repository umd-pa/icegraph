# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from icegraph.common.record import Array, ScalarList

__all__ = ["ColumnMetadata", "GlobalColumns", "SplitMap", "Stats"]


class ColumnMetadata(BaseModel):
    """Required data for each stored column."""

    # one entry per source column packed into the stored one
    names:  ScalarList[str]
    dtypes: ScalarList[str]

    # where each source column starts in the packed width, L + 1 of them
    offset: Array


class GlobalColumns(BaseModel):
    columns: dict[str, ColumnMetadata]


class SplitMap(BaseModel):
    splitmap: Array


class Stats(BaseModel):
    stats: dict[str, dict[str, Any]]
