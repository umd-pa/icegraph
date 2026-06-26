# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Generic, TypeVar, cast
from queue import Queue, Empty
from collections.abc import Iterator
from threading import Lock

__all__ = ["IterableQueue"]


O = TypeVar("O")  # output type

SENTINEL = object()


class IterableQueue(Iterator[O], Generic[O]):

    def __init__(self, *, maxsize: int = 10) -> None:
        # actual wrapped queue
        self._q: Queue[O | object] = Queue(maxsize)

        # active flag
        self._closed = False

        # lock
        self._lock = Lock()

    def __iter__(self) -> IterableQueue[O]:
        return self

    def __next__(self) -> O:
        item = self._q.get()
        if item is SENTINEL:
            raise StopIteration
        return cast(O, item)

    def put(self, item: O) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("Cannot put to a closed queue.")
        self._q.put(item)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._q.put(SENTINEL)
