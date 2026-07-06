# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import TYPE_CHECKING, Self
from dataclasses import dataclass

from icegraph.typing.common import ArrayF, ArrayI, ArrayB

from .factory import StatisticFactory
from .types import StatisticStruct, StatisticBundleStruct

if TYPE_CHECKING:
    from .statistic import Statistic

__all__ = ["StatisticBundle"]


@dataclass
class StatisticBundle:
    """Container for a coherent set of statistics computed on the same data."""
    stats: dict[str, Statistic]

    def get(self, kind: str) -> Statistic:
        """Return the statistic associated with the given kind."""
        stat = self.stats.get(kind)

        if stat is None:
            raise KeyError(f"No stat '{kind}' is registered in the statistic bundle.")

        return stat

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

        # perform the merge
        merged_stats: dict[str, Statistic] = {}
        for kind, stat in a.stats.items():
            merged_stats[kind] = type(stat).merge(a, b)

        return cls(merged_stats)

    def num_columns(self) -> int:
        """Return the number of columns tracked by the bundle."""
        if not self.stats:
            raise RuntimeError(
                f"{type(self).__name__}.num_columns: bundle contains no stats."
            )

        widths = {stat.num_columns() for stat in self.stats.values()}
        if len(widths) != 1:
            raise RuntimeError(
                f"{type(self).__name__}.num_columns: inconsistent column counts "
                f"across stats: {sorted(widths)}"
            )

        return widths.pop()

    def align_to(self, indices: ArrayI) -> Self:
        # validate permutation against the stats column count
        n = self.num_columns()
        if indices.shape != (n,):
            raise ValueError(
                f"{type(self).__name__}.align_to expected {n} indices, "
                f"got {indices.size}."
            )

        if set(indices.tolist()) != set(range(n)):
            raise ValueError(
                f"{type(self).__name__}.align_to indices must be a permutation "
                f"of 0..{n - 1}, got {sorted(indices.tolist())}."
            )

        for stat in self.stats.values():
            stat.align_to(indices)

        return self

    def filter_to(self, mask: ArrayB) -> Self:
        # filter each stat
        for stat in self.stats.values():
            stat.filter_to(mask)

        return self

    def to_struct(self) -> StatisticBundleStruct:
        """Serialize the bundle instance into a JSON-friendly dict structure."""
        return {
            "stats": {kind: stat.to_struct() for kind, stat in self.stats.items()}
        }

    @classmethod
    def from_struct(cls, struct: StatisticBundleStruct) -> Self:
        """Rebuild a StatisticBundle from a struct."""
        # make sure required keys are present
        if "stats" not in struct:
            raise KeyError("Param 'struct' missing required key: 'stats'")

        # unpack the bundle struct to individual stat structs
        stat_structs: dict[str, StatisticStruct] = struct["stats"]

        if not isinstance(stat_structs, dict):
            raise TypeError("Param 'struct[\"stats\"]' must be a dict.")

        # verify every array across all stats/spaces shares the same column count
        widths: set[int] = set()
        for kind, stat_struct in stat_structs.items():
            for space_value, array in stat_struct.items():
                widths.add(array.shape[-1])

        if len(widths) > 1:
            raise ValueError(
                f"{cls.__name__}.from_struct: inconsistent column counts across "
                f"stats/spaces: {sorted(widths)}"
            )

        # build stats from structs
        stats: dict[str, Statistic] = {}
        for kind, stat_struct in stat_structs.items():
            stats[kind] = StatisticFactory.from_struct(kind, stat_struct)

        return cls(stats)
