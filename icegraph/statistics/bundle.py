# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import TYPE_CHECKING, Self
from dataclasses import dataclass

import numpy as np

from icegraph.types.statistics import StatisticBundleStruct, StatisticStruct
from icegraph.types.common import ArrayF

from .factory import StatisticFactory

if TYPE_CHECKING:
    from .statistic import Statistic

__all__ = ["StatisticBundle"]


@dataclass
class StatisticBundle:
    """Container for a coherent set of statistics computed on the same data."""
    stats:      dict[str, Statistic]
    columns:    list[str]

    def __post_init__(self) -> None:
        if len(set(self.columns)) != len(self.columns):
            raise ValueError("Internal column list contains duplicates.")

    def get(self, kind: str) -> Statistic:
        """Return the statistic associated with the given kind."""
        return self.stats[kind]

    def compute(self, array: ArrayF) -> Self:
        """Run computation of stats for each stat in the bundle."""
        # compute all stats
        for stat in self.stats.values():
            stat.compute(array)

        return self

    @classmethod
    def merge(cls, a: StatisticBundle, b: StatisticBundle) -> Self:
        # make sure the stat bundles contain the same stat set
        diff = a.stats.keys() ^ b.stats.keys()
        if diff:
            raise ValueError(
                f"{cls.__name__} stats mismatch: {list(diff)}"
            )

        # make sure the stat bundles contain the same columns
        diff = sorted(set(a.columns) ^ set(b.columns))
        if diff:
            raise ValueError(
                f"{cls.__name__} columns mismatch: {list(diff)}"
            )

        # perform the merge
        merged_stats: dict[str, Statistic] = {}
        for kind, stat in a.stats.items():
            merged_stats[kind] = type(stat).merge(a, b)

        return cls(merged_stats, a.columns)

    def align_to(self, columns: list[str]) -> Self:
        # ensure all columns are unique
        if len(set(columns)) != len(columns):
            raise ValueError("Requested column list contains duplicates.")

        # ensure both have same members
        diff = sorted(set(self.columns) ^ set(columns))
        if diff:
            raise ValueError(
                f"{type(self).__name__}.align_to columns mismatch: {list(diff)}"
            )

        # build quick lookup dict and index array
        pos = {col: i for i, col in enumerate(self.columns)}
        indices = np.fromiter((pos[col] for col in columns), dtype=int)

        # align each stat array
        for stat in self.stats.values():
            stat.align_to(indices)

        # reorder columns to match
        self.columns[:] = columns

        return self

    def filter_to(self, columns: list[str]) -> Self:
        # coerce to sets for faster lookup, only really helpful for massive column lists
        current = set(self.columns)
        requested = set(columns)

        # raise if column set is empty
        if not requested:
            raise ValueError("Cannot filter to an empty column set.")

        # ensure all columns are unique
        if len(requested) != len(columns):
            raise ValueError("Requested column list contains duplicates.")

        # ensure requested columns exist
        missing = requested - current
        if missing:
            raise ValueError(f"Requested column list contains columns that do not exist: {sorted(missing)}")

        # build mask
        mask = np.fromiter((col in requested for col in self.columns), dtype=bool)

        # filter each stat array
        for stat in self.stats.values():
            stat.filter_to(mask)

        # filter columns to match, but do not reorder
        self.columns[:] = [c for c, keep in zip(self.columns, mask) if keep]

        return self

    def index_of(self, column: str) -> int:
        """
        Returns the index of ``column`` in the internal stat array.

        Args:
            column: Column name.
        """
        return self.columns.index(column)

    def to_struct(self) -> StatisticBundleStruct:
        """Serialize the bundle instance into a JSON-friendly dict structure."""
        return {
            "columns": self.columns,
            "stats": {kind: stat.to_struct() for kind, stat in self.stats.items()}
        }

    @classmethod
    def from_struct(cls, struct: StatisticBundleStruct) -> Self:
        """Rebuild a StatisticBundle from a struct."""
        # make sure required keys are present
        missing = [k for k in ("columns", "stats") if k not in struct]
        if missing:
            raise KeyError(f"Param 'struct' missing required keys: {missing}")

        # unpack the bundle struct to individual stat structs
        columns:        list[str]                   = struct["columns"]
        stat_structs:   dict[str, StatisticStruct]  = struct["stats"]

        if not isinstance(stat_structs, dict):
            raise TypeError("Param 'struct[\"stats\"]' must be a dict.")

        # build stats from structs
        stats: dict[str, Statistic] = {}
        for kind, stat_struct in stat_structs.items():
            stats[kind] = StatisticFactory.from_struct(kind, stat_struct)

        return cls(stats, columns)
