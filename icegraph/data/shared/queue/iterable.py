# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Generic, TypeVar, cast, Any
from queue import Queue, Empty
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
            self._size = ctx.Value("i", 0)
        else:
            self._q = Queue(maxsize)
            self._remaining = Value("i", producers)
            self._size = Value("i", 0)

        # 0 for unbounded
        self.maxsize = maxsize

        # active flag
        self._closed = False
        self._terminated = False

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
        return self._take(self._q.get())

    def poll(self, timeout: float) -> O | None:
        """
        Get the next item, or None if nothing arrived within `timeout`.

        Lets a consumer stay responsive to out-of-band shutdown signals instead
        of blocking forever on a queue no live producer will ever feed again.
        """
        try:
            item = self._q.get(timeout=timeout)
        except Empty:
            return None

        return self._take(item)

    def _take(self, item: O | _Sentinel) -> O:
        if isinstance(item, _Sentinel):
            raise StopIteration

        with self._size.get_lock():
            self._size.value -= 1
        return cast(O, item)

    def qsize(self) -> int:
        """Items waiting in the queue, not counting the sentinels that end it. Approximate while it is in use."""
        # a consumer can take an item before its producer counts it, briefly going below 0
        return max(0, self._size.value)

    def put(self, item: O) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("Cannot put to a closed queue.")
        self._q.put(item)

        # counted once queued, so a producer blocked on a full queue is not
        with self._size.get_lock():
            self._size.value += 1

    def done(self) -> None:
        """Called once per producer; closes the queue when the last producer finishes."""
        with self._lock:
            if self._terminated or self._remaining is None:
                return

        with self._remaining.get_lock():
            self._remaining.value -= 1
            last = self._remaining.value == 0
        if last:
            self.close()

    def close(self) -> None:
        with self._lock:
            if self._closed or self._terminated:
                return

            self._closed = True

        for _ in range(self._consumers):
            self._q.put(_Sentinel())

    def terminate(self) -> None:
        """Release OS-backed resources (mp queue feeder thread and named semaphores).
        Idempotent. Distinct from close(), which only broadcasts sentinels.
        """
        with self._lock:
            if self._terminated:
                return

            self._terminated = True

        if isinstance(self._q, MPQueueType):
            self._q.cancel_join_thread()  # don't wait on a feeder whose consumer is dead
            self._q.close()
        # stdlib Queue owns no OS handles
