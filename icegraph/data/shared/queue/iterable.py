# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Generic, TypeVar
from queue import Queue
from collections.abc import Iterator
from threading import Lock

__all__ = ["IterableQueue"]


O = TypeVar("O")  # output type

SENTINEL = object()


class IterableQueue(Iterator[O], Generic[O]):

    def __init__(self, *, maxsize: int = 5) -> None:
        # actual wrapped queue
        self._q: Queue[O | object] = Queue(maxsize)

        # active flag
        self._active = True

        # lock
        self._lock = Lock()

    def __iter__(self) -> IterableQueue[O]:
        return self

    def __next__(self) -> O:
        item = self._q.get()

        if item is SENTINEL:
            raise StopIteration

        return item

    def put(self, item: O) -> None:
        with self._lock:
            if not self._active:
                raise RuntimeError("Cannot put to an inactive queue.")
        self._q.put(item)

    def close(self) -> None:
        with self._lock:
            if not self._active:
                return
            self._active = False
        self._q.put(SENTINEL)
