# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from functools import cached_property
from typing import  Iterator, Any, ClassVar, overload
from bisect import bisect_right

import numpy as np

from icegraph.typing.common import ArrayI
from icegraph.common.record import Attributes, Record

from ...store import Store

from .cache import ShardLRUCache
from .config import LRUStoreConfig

__all__ = ["LRUShardStore"]


class LRUShardStore(Store[LRUStoreConfig]):
    name: ClassVar[str] = "lru-shard"
    version: ClassVar[int] = 1

    def build(self) -> None:
        return

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> LRUStoreConfig:
        return LRUStoreConfig(**config)

    @overload
    def __getitem__(self, index: int) -> Record: ...
    @overload
    def __getitem__(self, index: slice) -> list[Record]: ...
    @overload
    def __getitem__(self, index: int | slice) -> Record | list[Record]: ...

    def __getitem__(self, index: int | slice) -> Record | list[Record]:
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
        sample = reader[row_index]

        return sample

    def __len__(self) -> int:
        """Return the total number of records managed by the store."""
        return self._dataset_size

    @cached_property
    def _reader_cache(self) -> ShardLRUCache:
        return ShardLRUCache(self._ctx.readers, self.config.cache_size)

    @cached_property
    def _samples(self) -> ArrayI:
        return np.asarray([len(r) for r in self._ctx.readers])

    @cached_property
    def _offsets(self) -> ArrayI:
        return np.concatenate((np.array([0]), np.cumsum(self._samples)))  # (0 appended to start)

    @cached_property
    def _dataset_size(self) -> int:
        return int(np.sum(self._samples))

    @property
    def attrs(self) -> Iterator[Attributes]:
        """Iterate over all shard attributes in the dataset in a deterministic order."""
        for reader in self._reader_cache.iter_readers():
            yield reader.attrs

    def _index_from_global(self, index: int) -> tuple[int, int]:
        # binary search for shard index
        shard_index = bisect_right(self._offsets, index) - 1

        # get row within shard
        row_index = index - self._offsets[shard_index]

        # return as tuple (shard, row)
        return shard_index, row_index

    def close(self) -> None:
        """Ensure that all LMDB environments are closed when the reader is deleted."""
        self._reader_cache.clear()
