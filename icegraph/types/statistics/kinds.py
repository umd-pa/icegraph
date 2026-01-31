# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from enum import Enum

__all__ = ["StatisticKind"]


class StatisticKind(Enum):
    """Identifiers for supported statistics and count metrics."""
    TOTAL_COUNT = "total_count"
    NAN_COUNT = "nan_count"
    ZERO_COUNT = "zero_count"
    FINITE_COUNT = "finite_count"
    POSITIVE_COUNT = "positive_count"
    MEAN = "mean"
    M2 = "m2"
    MIN = "min"
    MAX = "max"

    @classmethod
    def all(cls) -> tuple[StatisticKind, ...]:
        return tuple(cls)
