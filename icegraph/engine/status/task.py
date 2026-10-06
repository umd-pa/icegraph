# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from .types import TaskState

if TYPE_CHECKING:
    from .status import Status

__all__ = ["Task"]


class Task:
    """
    One unit of work reported to the status.

    When used as a context manager the task finishes on exit, or fails if an exception
    is raised. Tasks opened on the same thread while inside it become its children.
    """

    def __init__(
            self,
            status: Status,
            name: str, *,
            parent: Task | None,
            total: float | None,
            detail: str,
            level: int
    ) -> None:
        self.name:      str                 = name
        self.parent:    Task | None         = parent
        self.depth:     int                 = 0 if parent is None else parent.depth + 1

        # progress, total is None when the amount of work is unknown
        self.total:     float | None        = total
        self.completed: float               = 0

        # what the task is doing right now (for example, the item being worked on)
        self.detail:    str                 = detail

        # level to log
        self.level:     int                 = level

        self.state:     TaskState           = TaskState.RUNNING
        self.error:     BaseException | None = None
        self.started:   float               = time.perf_counter()
        self.ended:     float | None        = None

        self._status = status

        # the thread stack this task was entered on
        self._stack:    list[Task] | None   = None

    @property
    def elapsed(self) -> float:
        end = self.ended if self.ended is not None else time.perf_counter()
        return end - self.started

    @property
    def running(self) -> bool:
        return self.state is TaskState.RUNNING

    def advance(self, n: float = 1) -> None:
        self.completed += n
        self._status._updated(self)

    def update(
            self, *,
            completed: float | None = None,
            total: float | None = None,
            detail: str | None = None
    ) -> None:
        if completed is not None:
            self.completed = completed

        if total is not None:
            self.total = total

        if detail is not None:
            self.detail = detail

        self._status._updated(self)

    def finish(self) -> None:
        self._status._end(self, TaskState.DONE)

    def fail(self, error: BaseException | None = None) -> None:
        self._status._end(self, TaskState.FAILED, error)

    def __enter__(self) -> Task:
        self._status._push(self)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._status._pop(self)

        if self.running:
            # a generator closed early stopped, it did not fail
            if exc is None or isinstance(exc, GeneratorExit):
                self.finish()
            else:
                self.fail(exc)

        return False

    def __repr__(self) -> str:
        progress = f", {self.completed:g}/{self.total:g}" if self.total else ""
        return f"Task({self.name!r}, {self.state.value}{progress}, {self.elapsed:.2f}s)"
