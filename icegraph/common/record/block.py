# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from icegraph.typing.common import ArrayI

__all__ = ["Column", "RecordBlock"]


@dataclass(frozen=True)
class Column:
    """One column of a record block, stored flat.

    ``values`` holds the rows of every record concatenated along dim 0.
    ``offsets`` maps record ``i`` to rows ``values[offsets[i]:offsets[i + 1]]``
    """

    values:  np.ndarray
    offsets: ArrayI

    @property
    def row_counts(self) -> ArrayI:
        """Rows contributed by each record."""
        return np.diff(self.offsets)


def _gather_rows(column: Column, order: ArrayI) -> tuple[np.ndarray, ArrayI]:
    """Gather ragged segments in ``order``, returning new flat values and offsets."""
    lengths = column.row_counts[order]

    out_off = np.zeros(len(order) + 1, dtype=np.int64)
    np.cumsum(lengths, out=out_off[1:])

    # rows of segment j land at out_off[j]: shift each output position back to its source row
    rows = np.arange(out_off[-1], dtype=np.int64) + np.repeat(column.offsets[order] - out_off[:-1], lengths)

    return column.values[rows], out_off


@dataclass(frozen=True)
class RecordBlock:
    """A batch of records in columnar form: flat value arrays plus row offsets."""

    height:  int
    columns: dict[str, Column]

    def take(self, indices: ArrayI) -> RecordBlock:
        """Select records by ``indices``."""
        columns = {
            name: Column(*_gather_rows(col, indices)) for name, col in self.columns.items()
        }
        return RecordBlock(height=len(indices), columns=columns)

    def permute(self, order: ArrayI) -> RecordBlock:
        """Reorder records by ``order`` (a permutation of ``range(height)``)."""
        if len(order) != self.height:
            raise ValueError(f"Expected a permutation of {self.height} records, got {len(order)}.")
        return self.take(order)

    def slice(self, start: int, stop: int) -> RecordBlock:
        """View of records ``[start, stop)``; values are numpy views, not copies."""
        columns: dict[str, Column] = {}
        for name, column in self.columns.items():
            lo, hi = column.offsets[start], column.offsets[stop]
            columns[name] = Column(
                column.values[lo:hi],
                column.offsets[start:stop + 1] - lo,
            )

        return RecordBlock(height=stop - start, columns=columns)

    @classmethod
    def concat(cls, blocks: Sequence[RecordBlock]) -> RecordBlock:
        """Concatenate blocks that share one schema, preserving block order."""
        if not blocks:
            raise ValueError("Cannot concatenate zero blocks.")

        if len(blocks) == 1:
            return blocks[0]

        names = blocks[0].columns.keys()
        if any(block.columns.keys() != names for block in blocks[1:]):
            raise ValueError("Cannot concatenate blocks with mismatched columns.")

        height = sum(block.height for block in blocks)

        columns: dict[str, Column] = {}
        for name in names:
            parts = [block.columns[name] for block in blocks]
            values = np.concatenate([part.values for part in parts], axis=0)

            # re-base each block's offsets onto the concatenated values
            offsets = np.zeros(height + 1, dtype=np.int64)
            base, row = 0, 0
            for block, part in zip(blocks, parts):
                offsets[row + 1:row + block.height + 1] = part.offsets[1:] + base
                base += part.offsets[-1]
                row += block.height

            columns[name] = Column(values, offsets)

        return cls(height=height, columns=columns)
