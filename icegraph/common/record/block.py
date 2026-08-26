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
    ``offsets`` maps record ``i`` to rows ``values[offsets[i]:offsets[i + 1]]``;
    ``None`` means exactly one row per record (``values[i]``).
    """

    values:  np.ndarray
    offsets: ArrayI | None = None

    def row_counts(self, height: int) -> ArrayI:
        """Rows contributed by each record."""
        if self.offsets is None:
            return np.ones(height, dtype=np.int64)

        return np.diff(self.offsets)


def _gather_rows(values: np.ndarray, offsets: ArrayI, order: ArrayI) -> tuple[np.ndarray, ArrayI]:
    """Gather ragged segments in ``order``, returning new flat values and offsets."""
    lengths = np.diff(offsets)[order]

    out_off = np.zeros(len(order) + 1, dtype=np.int64)
    np.cumsum(lengths, out=out_off[1:])

    # rows of segment j land at out_off[j]: shift each output position back to its source row
    rows = np.arange(out_off[-1], dtype=np.int64) + np.repeat(offsets[order] - out_off[:-1], lengths)

    return values[rows], out_off


@dataclass(frozen=True)
class RecordBlock:
    """A batch of records in columnar form: flat value arrays plus row offsets.

    This is the unit the read/assemble path works in; records are never
    materialized individually.
    """

    height:  int
    columns: dict[str, Column]

    def permute(self, order: ArrayI) -> RecordBlock:
        """Reorder records by ``order`` (a permutation of ``range(height)``)."""
        columns: dict[str, Column] = {}
        for name, column in self.columns.items():
            if column.offsets is None:
                columns[name] = Column(column.values[order])
            else:
                columns[name] = Column(*_gather_rows(column.values, column.offsets, order))

        return RecordBlock(height=len(order), columns=columns)

    def slice(self, start: int, stop: int) -> RecordBlock:
        """View of records ``[start, stop)``; values are numpy views, not copies."""
        columns: dict[str, Column] = {}
        for name, column in self.columns.items():
            if column.offsets is None:
                columns[name] = Column(column.values[start:stop])
            else:
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
            ragged = [part.offsets is not None for part in parts]

            if any(ragged) != all(ragged):
                raise ValueError(f"Column {name!r} is ragged in some blocks but not others.")

            values = np.concatenate([part.values for part in parts], axis=0)

            if not any(ragged):
                columns[name] = Column(values)
                continue

            # re-base each block's offsets onto the concatenated values
            offsets = np.zeros(height + 1, dtype=np.int64)
            base, row = 0, 0
            for block, part in zip(blocks, parts):
                assert part.offsets is not None
                offsets[row + 1:row + block.height + 1] = part.offsets[1:] + base
                base += part.offsets[-1]
                row += block.height

            columns[name] = Column(values, offsets)

        return cls(height=height, columns=columns)
