# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Generic, TypeVar, cast, Any
from queue import Queue
from multiprocessing import Value, get_context
from multiprocessing.context import BaseContext
from multiprocessing.queues import Queue as MPQueueType
from collections.abc import Iterator
from threading import Lock

__all__ = ["IterableQueue"]


O = TypeVar("O")  # output type


class _Sentinel:
    pass


class IterableQueue(Iterator[O], Generic[O]):

    def __init__(
        self, *,
        mp: bool = False,
        maxsize: int = 10,
        producers: int = 1,
        consumers: int = 1,
        ctx: BaseContext | None = None
    ) -> None:
        # actual wrapped queue
        # polars is not fork-safe, so mp queues must come from a spawn context
        # matching the processes they connect
        self._q: Queue[O | _Sentinel] | MPQueueType[O | _Sentinel]
        if mp:
            ctx = ctx if ctx is not None else get_context("spawn")
            self._q = ctx.Queue(maxsize)
            self._remaining = ctx.Value("i", producers)
        else:
            self._q = Queue(maxsize)
            self._remaining = Value("i", producers)

        # active flag
        self._closed = False

        # multiprocessing
        self._consumers = consumers

        # lock
        self._lock = Lock()

    def __getstate__(self) -> dict[str, Any]:
        # thread locks cannot cross process boundaries, recreated on unpickle
        state = self.__dict__.copy()
        state.pop("_lock", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._lock = Lock()

    def __iter__(self) -> IterableQueue[O]:
        return self

    def __next__(self) -> O:
        item = self._q.get()
        if isinstance(item, _Sentinel):
            raise StopIteration
        return cast(O, item)

    def put(self, item: O) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("Cannot put to a closed queue.")
        self._q.put(item)

    def done(self) -> None:
        """Called once per producer; closes the queue when the last producer finishes."""
        with self._remaining.get_lock():
            self._remaining.value -= 1
            last = self._remaining.value == 0
        if last:
            self.close()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return

            self._closed = True

        for _ in range(self._consumers):
            self._q.put(_Sentinel())
