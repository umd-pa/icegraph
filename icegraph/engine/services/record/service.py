# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Iterator, Any, ClassVar, overload
from functools import cached_property
from operator import attrgetter

from icegraph.common.files import Source
from icegraph.common.record import Record, Attributes, GlobalAttributes

from ..service import Service

from .config import RecordConfig
from .store import Store, StoreFactory, StoreContext
from .reader import Reader, ReaderFactory, ReaderContext

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
    def __getitem__(self, index: slice) -> list[Record]: ...
    @overload
    def __getitem__(self, index: int | slice) -> Record | list[Record]: ...

    def __getitem__(self, index: int | slice) -> Record | list[Record]:
        return self._store[index]

    def __len__(self) -> int:
        return len(self._store)

    @cached_property
    def _target_file_ext(self) -> str:
        reader_cls = ReaderFactory.get_class(self.config.reader.name)
        return reader_cls.file_ext

    @cached_property
    def file_count(self) -> int:
        return len(list(self.source.resolve(self._target_file_ext)))

    @cached_property
    def source(self) -> Source:
        return Source(self.config.source)

    @cached_property
    def _readers(self) -> list[Reader]:
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

        return readers

    @cached_property
    def _store(self) -> Store:
        # build the store
        store = StoreFactory.create(self.config.store.name, **self.config.store.kwargs)

        # attach the store
        ctx = StoreContext(readers=self._readers, ignore_checksum=self.config.ignore_checksum)
        store.attach(ctx)

        return store

    def attrs(self) -> Iterator[Attributes]:
        return self._store.attrs

    @property
    def global_attrs(self) -> GlobalAttributes:
        return self._store.global_attrs
