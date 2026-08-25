# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Iterator, Any, ClassVar, overload
from functools import cached_property
from operator import itemgetter, attrgetter
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from rich.progress import track, Progress

from icegraph.common.files import Source
from icegraph.common.record import Record, Attributes, GlobalAttributes
from icegraph.typing.common import ArrayI
from icegraph.ui import console

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

    def __iter__(self) -> Iterator[Record]:
        """Iterate through all records."""
        for i in range(len(self)):
            yield self[i]

    @overload
    def __getitem__(self, index: int) -> Record: ...

    @overload
    def __getitem__(self, index: slice | list[int] | ArrayI) -> list[Record]: ...

    @overload
    def __getitem__(self, index: int | slice | list[int] | ArrayI) -> Record | list[Record]: ...

    def __getitem__(self, index: int | slice | list[int] | ArrayI) -> Record | list[Record]:
        """
        Retrieve one or more records.

        Args:
            index: Integer index or slice object.
        """
        scalar_input = False
        if isinstance(index, (int, np.integer)) and not isinstance(index, bool):
            scalar_input = True

            # check index valid
            if not (-len(self) <= index < len(self)):
                raise IndexError(f"Index {index} out of bounds for dataset of length {len(self)}.")

            index_array = np.asarray([index])

        elif isinstance(index, slice):
            if index.start is not None and not (-len(self) <= index.start <= len(self)):
                raise IndexError(f"Index slice start {index.start} out of bounds for dataset of length {len(self)}")

            if index.stop is not None and not (-len(self) <= index.stop <= len(self)):
                raise IndexError(f"Index slice stop {index.stop} out of bounds for dataset of length {len(self)}")

            if index.step == 0:
                raise ValueError("Index slice step cannot be zero")

            index_array = np.asarray(list(range(*index.indices(len(self)))))

        elif isinstance(index, (list, np.ndarray)):
            index_array = np.asarray(index)

            if index_array.ndim > 1:
                raise ValueError(f"Index array cannot contain more than 1 dim, got {index_array.ndim}.")

            if not np.issubdtype(index_array.dtype, np.integer):
                raise TypeError(f"Index array must have an integer dtype, got {index_array.dtype}.")

            out_of_bounds = (index_array < -len(self)) | (index_array >= len(self))

            if out_of_bounds.any():
                raise IndexError(f"Index {index_array[out_of_bounds]} out of bounds for dataset of length {len(self)}.")

        # assert index is an int, slice, list, ndarray
        else:
            raise TypeError(f"Index must be of type int | list[int] | np.NDArray[np.integer] | slice, got '{type(index).__name__}'.")

        # at this point, index should be an array
        assert isinstance(index_array, np.ndarray)

        # wrap
        index_array = index_array % len(self)

        # get shard and row indices and load reader from cache
        shard_idxs, row_idxs = self._indices_from_global(index_array)

        positions: list[ArrayI] = []
        gathered: list[Record] = []

        for shard_idx in np.unique(shard_idxs):
            mask = shard_idxs == shard_idx
            reader = self._cache.get_reader(shard_idx)

            positions.append(np.flatnonzero(mask))
            gathered.extend(reader[row_idxs[mask]])

        if not positions:
            return []

        # restore requested order
        slots = np.concatenate(positions)
        inverse = np.empty_like(slots)
        inverse[slots] = np.arange(slots.size)

        samples = [gathered[i] for i in inverse]

        return samples[0] if scalar_input else samples

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
