# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Iterator, Any, ClassVar
from collections.abc import Collection
from functools import cached_property
from operator import attrgetter

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

    # built by setup()
    _cache:         ShardLRUCache
    _offsets:       ArrayI
    record_count:   int
    global_attrs:   GlobalAttributes

    def build(self) -> None:
        return

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> RecordConfig:
        return RecordConfig(**config)

    def setup(self) -> None:
        # open every shard once, which is where the dataset startup time mostly goes
        self._cache = self._build_cache()

        samples = np.asarray([len(r) for r in self._cache.iter_readers()])
        self._offsets = np.concatenate((np.array([0]), np.cumsum(samples)))  # (0 appended to start)
        self.record_count = int(np.sum(samples))

        self.global_attrs = GlobalAttributes.from_attrs(self.attrs(), ignore_checksum=self.config.ignore_checksum)

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
    def file_count(self) -> int:
        file_count = len(list(self.source.resolve(self._target_file_ext)))

        # do a quick check to ensure non-0 file count
        if file_count == 0:
            raise FileNotFoundError("Record service received 0 data files.")

        return file_count

    def _indices_from_global(self, indices: ArrayI) -> tuple[ArrayI, ArrayI]:
        shard_indices = np.searchsorted(self._offsets, indices, side="right") - 1
        row_indices = indices - self._offsets[shard_indices]

        return shard_indices, row_indices

    @cached_property
    def source(self) -> Source:
        return Source(self.config.source)

    def _build_cache(self) -> ShardLRUCache:
        paths = list(self.source.resolve(self._target_file_ext))

        readers: list[Reader] = []
        for path in self._ctx.status.track(paths, "Indexing shards"):
            # create the reader
            reader = ReaderFactory.create(self.config.reader.name, **self.config.reader.kwargs)

            # attach the reader given specific file path
            ctx = ReaderContext(path=path)
            reader.attach(ctx)

            # attributes are read here so the shards can be sorted by id below, this is the
            # expensive part since each files metadata has to be opened
            _ = reader.attrs
            reader.close()

            # append to list
            readers.append(reader)

        # sort readers by shard id
        readers.sort(key=attrgetter("attrs.shard_id"))

        return ShardLRUCache(readers, self.config.cache_size)

    @cached_property
    def _target_file_ext(self) -> str:
        reader_cls = ReaderFactory.get_class(self.config.reader.name)
        return reader_cls.file_ext

    def attrs(self) -> Iterator[Attributes]:
        """Iterate over all shard attributes in the dataset in a deterministic order."""
        for reader in self._cache.iter_readers():
            yield reader.attrs
