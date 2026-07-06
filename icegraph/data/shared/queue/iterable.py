# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Generic, TypeVar, cast
from queue import Queue
from multiprocessing import Queue as MPQueue, Value
from multiprocessing.queues import Queue as MPQueueType
from collections.abc import Iterator
from threading import Lock

__all__ = ["IterableQueue"]


O = TypeVar("O")  # output type


class _Sentinel:
    pass


class IterableQueue(Iterator[O], Generic[O]):

    def __init__(self, *, mp: bool = False, maxsize: int = 10, producers: int = 1, consumers: int = 1) -> None:
        # actual wrapped queue
        self._q: Queue[O | _Sentinel] | MPQueueType[O | _Sentinel] = Queue(maxsize) if mp is False else MPQueue(maxsize)

        # active flag
        self._closed = False

        # multiprocessing
        self._consumers = consumers
        self._remaining = Value("i", producers)

        # lock
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
