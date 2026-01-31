# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import  Iterator, Self
from bisect import bisect_right
from operator import attrgetter

import numpy as np

from icegraph.types.files import Source
from icegraph.types.common import ArrayI, ArrayG
from icegraph.types.data import AttributeDomain

from .reader import Reader
from .cache import ShardLRUCache
from ..types import Attributes, GlobalAttributes

__all__ = ["ShardStore"]


class ShardStore:
    """Provides access to one or more files from the same dataset."""

    def __init__(self, reader: type[Reader], source: Source, cache_size: int = 32) -> None:
        """Initialize the shard store."""
        self.source: Source = source

        # initialize readers and sort by shard id (for deterministic behavior)
        readers: list[Reader] = [reader(path) for path in source.resolve(reader.file_ext)]
        readers.sort(key=attrgetter("attrs.shard_id"))

        # cache shard ids
        self._shard_ids: list[str] = [r.attrs.shard_id for r in readers]

        # initialize LRU cache
        self._reader_cache = ShardLRUCache(readers, cache_size)

        # get file sample counts and cache cumulative sum for bisection
        self._samples: ArrayI = np.asarray([len(r) for r in readers])
        self._offsets: ArrayI = np.concatenate((np.array([0]), np.cumsum(self._samples)))  # (0 appended to start)

        # cache total dataset size
        self._dataset_size: int = int(np.sum(self._samples))

        # cache for global dataset attributes
        self._global_attrs: GlobalAttributes | None = None

    def __getitem__(self, index: int | slice) -> dict[str, ArrayG] | list[dict[str, ArrayG]]:
        """
        Retrieve one record or a slice of records.

        Args:
            index: Integer index or slice object.
        """
        # handle slices
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            return [self[i] for i in range(start, stop, step)]

        # assert index is an int
        if not isinstance(index, int):
            raise TypeError(f"Index must be of type int, got '{type(index).__name__}'.")

        # assert index is valid
        if not (-len(self) <= index < len(self)):
            raise IndexError(f"Index {index} out of bounds for dataset of length {len(self)}.")

        # normalize index
        index = index % len(self)

        # get shard and row indices and load reader from cache
        shard_index, row_index = self._index_from_global(index)
        reader = self._reader_cache.get_reader(shard_index)

        # get from the reader
        return reader[row_index]

    def __len__(self) -> int:
        """Return the total number of records managed by the store."""
        return self._dataset_size

    def __iter__(self) -> Iterator[dict[str, ArrayG]]:
        """Iterate through all records."""
        for i in range(len(self)):
            yield self[i]

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def _index_from_global(self, index: int) -> tuple[int, int]:
        # binary search for shard index
        shard_index = bisect_right(self._offsets, index) - 1

        # get row within shard
        row_index = index - self._offsets[shard_index]

        # return as tuple (shard, row)
        return shard_index, row_index

    def _build_global_attrs(self, checksum: str | None = None) -> GlobalAttributes:
        # get iterator over attributes
        it = iter(self.attrs)

        try:
            first_attr = next(it)
        except StopIteration:
            raise RuntimeError("Cannot build global attributes for an empty dataset.")

        expected = checksum if checksum is not None else first_attr.checksum

        # verify each checksum is identical to the first (or to the supplied checksum)
        if first_attr.checksum != expected:
            raise ValueError(f"Checksums do not match across shards; expected {expected}, got {first_attr.checksum}.")

        for attr in it:
            if attr.checksum != expected:
                raise ValueError(f"Checksums do not match across shards; expected {expected}, got {attr.checksum}.")

        # because each checksum is identical, just grab globals from first attribute
        return GlobalAttributes(first_attr.get(AttributeDomain.GLOBAL))

    def global_attrs(self, *, checksum: str | None = None) -> GlobalAttributes:
        if self._global_attrs is None:
            self._global_attrs = self._build_global_attrs(checksum)
        return self._global_attrs

    @property
    def attrs(self) -> Iterator[Attributes]:
        """Iterate over all shard attributes in the dataset in a deterministic order."""
        for reader in self._reader_cache.iter_readers():
            yield reader.attrs

    def close(self) -> None:
        """Ensure that all LMDB environments are closed when the reader is deleted."""
        self._reader_cache.clear()
