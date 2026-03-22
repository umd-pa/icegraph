# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import  Iterator, Any, ClassVar
from bisect import bisect_right
from operator import attrgetter

import numpy as np

from icegraph.types.common import ArrayI
from icegraph.types.data import AttributeDomain
from icegraph.trainer.services.data.types import Attributes, GlobalAttributes
from icegraph.types.files import Source

from ...readers import Reader, ReaderFactory, ReaderContext
from ...store import Store

from .cache import ShardLRUCache
from .config import Config

__all__ = ["LRUShardStore"]


class LRUShardStore(Store[Config]):
    name: ClassVar[str] = "lru-shard"
    version: ClassVar[int] = 1

    _shard_ids: list[str]
    _reader_cache: ShardLRUCache
    _samples: ArrayI
    _offsets: ArrayI
    _dataset_size: int
    _global_attrs: GlobalAttributes | None

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> Config:
        return Config(**config)

    def build(self) -> None:
        # cache for global dataset attributes
        self._global_attrs = None

    def on_attach(self) -> None:
        # build the readers
        reader_config = self.config.reader

        # first build a single reader to determine file extension
        sample = ReaderFactory.create(reader_config.name, **reader_config.kwargs)
        file_ext = type(sample).file_ext

        readers: list[Reader] = []
        for path in self._ctx.source.resolve(file_ext):
            # create the reader
            reader = ReaderFactory.create(reader_config.name, **reader_config.kwargs)

            # attach the reader given specific file path
            ctx = ReaderContext(path=path)
            reader.attach(ctx)

            # append to list
            readers.append(reader)

        # sort readers by shard id
        readers.sort(key=attrgetter("attrs.shard_id"))

        # cache shard ids
        self._shard_ids = [r.attrs.shard_id for r in readers]

        # initialize LRU cache
        self._reader_cache = ShardLRUCache(readers, self.config.cache_size)

        # get file sample counts and cache cumulative sum for bisection
        self._samples = np.asarray([len(r) for r in readers])
        self._offsets = np.concatenate((np.array([0]), np.cumsum(self._samples)))  # (0 appended to start)

        # cache total dataset size
        self._dataset_size = int(np.sum(self._samples))

    def __getitem__(self, index: int | slice) -> dict[str, Any] | list[dict[str, Any]]:
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

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate through all records."""
        for i in range(len(self)):
            yield self[i]

    def _index_from_global(self, index: int) -> tuple[int, int]:
        # binary search for shard index
        shard_index = bisect_right(self._offsets, index) - 1

        # get row within shard
        row_index = index - self._offsets[shard_index]

        # return as tuple (shard, row)
        return shard_index, row_index

    def _build_global_attrs(self) -> GlobalAttributes:
        # get iterator over attributes
        it = iter(self.attrs)

        try:
            first_attr = next(it)
        except StopIteration:
            raise RuntimeError("Cannot build global attributes for an empty dataset.")

        # verify each checksum is identical to the first
        for attr in it:
            if attr.checksum != first_attr.checksum:
                raise ValueError(f"Checksums do not match across shards; expected {first_attr.checksum}, got {attr.checksum}.")

        # because each checksum is identical, just grab globals from first attribute
        return GlobalAttributes(first_attr.get(AttributeDomain.GLOBAL))

    @property
    def global_attrs(self) -> GlobalAttributes:
        if self._global_attrs is None:
            self._global_attrs = self._build_global_attrs()
        return self._global_attrs

    @property
    def attrs(self) -> Iterator[Attributes]:
        """Iterate over all shard attributes in the dataset in a deterministic order."""
        for reader in self._reader_cache.iter_readers():
            yield reader.attrs

    def close(self) -> None:
        """Ensure that all LMDB environments are closed when the reader is deleted."""
        self._reader_cache.clear()
