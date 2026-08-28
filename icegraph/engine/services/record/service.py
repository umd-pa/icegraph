# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Iterator, Any, ClassVar
from collections.abc import Collection
from functools import cached_property
from operator import attrgetter
import time

import numpy as np

from icegraph.common.files import Source
from icegraph.common.record import RecordBlock, Attributes, GlobalAttributes
from icegraph.typing.common import ArrayI

from ..service import Service

from .config import RecordConfig
from .reader import Reader, ReaderFactory, ReaderContext
from .cache import ShardLRUCache

import logging
logger = logging.getLogger(__name__)

__all__ = ["RecordService"]


class RecordService(Service[RecordConfig]):
    name: ClassVar[str] = "record"
    version: ClassVar[int] = 1

    def build(self) -> None:
        return

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> RecordConfig:
        return RecordConfig(**config)

    def read(self, indices: ArrayI, columns: Collection[str] | None = None) -> RecordBlock:
        """
        Read the given records as one columnar block.

        Indices must be ascending and in bounds; shards are visited in order,
        so the block preserves the requested order.

        ``columns`` restricts the read to the named columns; ``None`` reads all.
        """
        if not isinstance(indices, np.ndarray):
            raise TypeError(f"Indices must be an npt.NDArray, got {type(indices).__name__}.")

        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
            raise TypeError(f"Indices must be a 1-dim integer array, got ndim {indices.ndim}, dtype {indices.dtype}.")

        if len(indices) == 0:
            raise ValueError("Cannot read an empty index array.")

        if indices[0] < 0 or indices[-1] >= len(self) or np.any(np.diff(indices) <= 0):
            raise IndexError(
                f"Indices must be strictly ascending and within [0, {len(self)}), "
                f"got range [{indices[0]}, {indices[-1]}]."
            )

        # get shard and row indices and load reader from cache
        shard_idxs, row_idxs = self._indices_from_global(indices)

        blocks: list[RecordBlock] = []
        for shard_idx in np.unique(shard_idxs):
            reader = self._cache.get_reader(shard_idx)
            blocks.append(reader.read(row_idxs[shard_idxs == shard_idx], columns))

        return RecordBlock.concat(blocks)

    def __len__(self) -> int:
        """Return the total number of records managed by the service."""
        return self.record_count

    @cached_property
    def record_count(self) -> int:
        return int(np.sum(self._samples))

    @cached_property
    def file_count(self) -> int:
        file_count = len(list(self.source.resolve(self._target_file_ext)))

        # do a quick check to ensure non-0 file count
        if file_count == 0:
            raise FileNotFoundError("Record service received 0 data files.")

        return file_count

    @cached_property
    def _samples(self) -> ArrayI:
        start = time.perf_counter()
        samples = np.asarray([len(r) for r in self._cache.iter_readers()])
        logger.info(f"[RecordService] Loaded sample counts in {time.perf_counter() - start} s.")
        return samples

    @cached_property
    def _offsets(self) -> ArrayI:
        return np.concatenate((np.array([0]), np.cumsum(self._samples)))  # (0 appended to start)

    def _indices_from_global(self, indices: ArrayI) -> tuple[ArrayI, ArrayI]:
        shard_indices = np.searchsorted(self._offsets, indices, side="right") - 1
        row_indices = indices - self._offsets[shard_indices]

        return shard_indices, row_indices

    @cached_property
    def source(self) -> Source:
        return Source(self.config.source)

    @cached_property
    def _cache(self) -> ShardLRUCache:
        start = time.perf_counter()
        readers: list[Reader] = []
        for path in self.source.resolve(self._target_file_ext):
            # create the reader
            reader = ReaderFactory.create(self.config.reader.name, **self.config.reader.kwargs)

            # attach the reader given specific file path
            ctx = ReaderContext(path=path)
            reader.attach(ctx)

            # append to list
            readers.append(reader)

        # sort readers by shard id
        readers.sort(key=attrgetter("attrs.shard_id"))

        # since we are sorting by shard id, we need to open each file handle to access attributes
        # thus we need to go through and close all readers again
        for reader in readers:
            reader.close()

        cache = ShardLRUCache(readers, self.config.cache_size)
        logger.info(f"[RecordService] Built shard cache in {time.perf_counter() - start} s.")
        return cache

    @cached_property
    def _target_file_ext(self) -> str:
        reader_cls = ReaderFactory.get_class(self.config.reader.name)
        return reader_cls.file_ext

    def attrs(self) -> Iterator[Attributes]:
        """Iterate over all shard attributes in the dataset in a deterministic order."""
        for reader in self._cache.iter_readers():
            yield reader.attrs

    @cached_property
    def global_attrs(self) -> GlobalAttributes:
        start = time.perf_counter()
        gattrs = GlobalAttributes.from_attrs(self.attrs(), ignore_checksum=self.config.ignore_checksum)
        logger.info(f"[RecordService] Loaded global attrs in {time.perf_counter() - start} s.")
        return gattrs
