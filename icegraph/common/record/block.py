# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

import ctypes
import ctypes.util
import sys
from dataclasses import dataclass
from functools import lru_cache
from math import ceil
from typing import Callable, ClassVar, Sequence

import numpy as np

from icegraph.typing.common import ArrayI

__all__ = ["Column", "RecordBlock", "PoolBuffer"]


@dataclass(frozen=True)
class Column:
    """One column of a record block, stored flat.

    ``values`` holds the rows of every record concatenated along dim 0.
    ``offsets`` maps record ``i`` to rows ``values[offsets[i]:offsets[i + 1]]``
    """

    values:  np.ndarray
    offsets: ArrayI

    @property
    def rows(self) -> int:
        return int(self.offsets[-1])

    @property
    def row_counts(self) -> ArrayI:
        """Rows contributed by each record."""
        return np.diff(self.offsets)


def _gather_rows(column: Column, order: ArrayI, offsets: ArrayI | None = None) -> tuple[ArrayI, ArrayI]:
    """Plan a ragged gather of ``column`` in ``order``.

    Fills ``offsets`` for the selection and returns the source rows to read in
    output order. The caller owns where the values land, so this serves both a
    fresh allocation and a reused buffer. ``offsets`` is allocated when omitted.
    """
    lengths = column.row_counts[order]

    if offsets is None:
        offsets = np.empty(len(order) + 1, dtype=np.int64)

    offsets[0] = 0
    np.cumsum(lengths, out=offsets[1:])

    # rows of segment j land at offsets[j]: shift each output position back to its source row
    rows = np.arange(offsets[-1], dtype=np.int64) + np.repeat(column.offsets[order] - offsets[:-1], lengths)

    return rows, offsets

@dataclass(frozen=True)
class RecordBlock:
    """A batch of records in columnar form: flat value arrays plus row offsets."""

    height:  int
    columns: dict[str, Column]

    def take(self, indices: ArrayI) -> RecordBlock:
        """Select records by ``indices``."""
        columns: dict[str, Column] = {}
        for name, column in self.columns.items():
            rows, offsets = _gather_rows(column, indices)
            columns[name] = Column(column.values[rows], offsets)

        return RecordBlock(height=len(indices), columns=columns)

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

            # rebase each blocks offsets onto the concatenated values
            offsets = np.zeros(height + 1, dtype=np.int64)
            base, row = 0, 0
            for block, part in zip(blocks, parts):
                offsets[row + 1:row + block.height + 1] = part.offsets[1:] + base
                base += part.offsets[-1]
                row += block.height

            columns[name] = Column(values, offsets)

        return cls(height=height, columns=columns)


class PoolBuffer:
    """Reusable backing store for concatenating blocks into one pool.

    Concatenating a fresh pool on every refill allocates and frees tens of MB per
    worker, which the kernel pays for in page-zeroing and TLB shootdowns rather
    than useful work. Holding one buffer per column keeps the contiguous copy while
    dropping churn.

    The returned block views the buffer, so it stays valid only until the next
    ``concat`` on the same instance. Everything derived from it must be a copy.
    Not thread safe.
    """

    # spare capacity kept when growing, so ragged pools stop reallocating quickly
    _HEADROOM: ClassVar[float] = 1.25

    def __init__(self) -> None:
        self._values: dict[str, np.ndarray] = {}
        self._offsets: dict[str, ArrayI] = {}

    def _values_for(self, name: str, rows: int, template: np.ndarray) -> np.ndarray:
        buffer = self._values.get(name)

        if (
            buffer is None
            or buffer.shape[0] < rows
            or buffer.shape[1:] != template.shape[1:]  # this should never be the case, just for safety
            or buffer.dtype != template.dtype  # also should never be the case
        ):
            capacity = ceil(rows * self._HEADROOM)
            buffer = self._values[name] = np.empty((capacity,) + template.shape[1:], dtype=template.dtype)

        return buffer[:rows]

    def _offsets_for(self, name: str, height: int) -> ArrayI:
        buffer = self._offsets.get(name)
        needed = height + 1

        if buffer is None or len(buffer) < needed:
            buffer = self._offsets[name] = np.empty(ceil(needed * self._HEADROOM), dtype=np.int64)

        return buffer[:needed]

    def _owns(self, block: RecordBlock) -> bool:
        """Whether any of ``block``'s storage lives in this buffer."""
        stores = (*self._values.values(), *self._offsets.values())

        return any(
            np.may_share_memory(column.values, store) or np.may_share_memory(column.offsets, store)
            for column in block.columns.values()
            for store in stores
        )

    def take(self, block: RecordBlock, indices: ArrayI) -> RecordBlock:
        """Select records out of ``block`` into this buffer.

        The same selection as ``RecordBlock.take``, gathered straight into reused
        storage rather than a fresh allocation. Use a buffer that ``block`` does
        not come from.

        Falls back to allocating when ``block`` does view this buffer, since the
        gather would otherwise read rows it had already overwritten.
        """
        if self._owns(block):
            return block.take(indices)

        columns: dict[str, Column] = {}
        for name, column in block.columns.items():
            rows, offsets = _gather_rows(column, indices, self._offsets_for(name, len(indices)))

            values = self._values_for(name, int(offsets[-1]), column.values)
            np.take(column.values, rows, axis=0, out=values)

            columns[name] = Column(values, offsets)

        return RecordBlock(height=len(indices), columns=columns)

    def concat(self, blocks: Sequence[RecordBlock]) -> RecordBlock:
        """Concatenate into the buffer, invalidating any block returned earlier."""
        if not blocks:
            raise ValueError("Cannot concatenate zero blocks.")

        # a lone block is already contiguous, and copying it in would only add work
        if len(blocks) == 1:
            return blocks[0]

        names = blocks[0].columns.keys()
        if any(block.columns.keys() != names for block in blocks[1:]):
            raise ValueError("Cannot concatenate blocks with mismatched columns.")

        height = sum(block.height for block in blocks)

        columns: dict[str, Column] = {}
        for name in names:
            parts = [block.columns[name] for block in blocks]
            rows = sum(part.rows for part in parts)

            # retrieve reused buffers to populate
            values = self._values_for(name, rows, parts[0].values)
            offsets = self._offsets_for(name, height)
            offsets[0] = 0

            # copy each blocks rows in
            record = row = 0
            for block, part in zip(blocks, parts):
                values[row:row + part.rows] = part.values
                offsets[record + 1:record + block.height + 1] = part.offsets[1:] + row

                row += part.rows
                record += block.height

            columns[name] = Column(values, offsets)

        return RecordBlock(height=height, columns=columns)
