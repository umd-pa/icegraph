# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Iterable, Iterator, Sized
from typing import Any, TypeVar

from .config import StatusConfig
from .task import Task
from .throughput import Throughput
from .types import TaskState, EventKind, TaskEvent, LogEntry, LogEvent, StatusEvent, Listener

logger = logging.getLogger(__name__)

__all__ = ["Status"]


T = TypeVar("T")

# every icegraph logger propagates to this one
_PACKAGE_LOGGER: str = "icegraph"


class _LogHandler(logging.Handler):
    """Routes every record logged under the package into the status."""

    def __init__(self, status: Status) -> None:
        super().__init__()
        self._status = status
        self._local = threading.local()

    def emit(self, record: logging.LogRecord) -> None:
        # a listener that logs while handling a record would otherwise loop forever
        if getattr(self._local, "busy", False):
            return

        self._local.busy = True
        try:
            self._status._log(record)
        finally:
            self._local.busy = False


class Status:
    """
    Central record of what the run is doing.

    Code anywhere in the backend opens tasks here to report the work it is doing and
    its progress. Every record logged under the package is routed through it too,
    before going on to the configured logging handlers as usual.

    Listeners can subscribe to the stream of task and log events.
    """

    _lock:          threading.RLock
    _local:         threading.local
    _active:        list[Task]
    _logs:          deque[LogEntry]
    _listeners:     list[Listener]
    _throughputs:   dict[str, Throughput]
    _handler:       _LogHandler | None

    def __init__(self, config: StatusConfig) -> None:
        self.config: StatusConfig = config
        self._reset()

        # route package logging through here
        self._handler = _LogHandler(self)
        assert self._handler is not None
        logging.getLogger(_PACKAGE_LOGGER).addHandler(self._handler)

    def _reset(self) -> None:
        self._lock = threading.RLock()
        self._local = threading.local()
        self._active = []
        self._logs = deque(maxlen=self.config.log_history)
        self._listeners = []
        self._throughputs = {}
        self._handler = None

    def __getstate__(self) -> dict[str, Any]:
        # listeners hold live displays and nothing here is shared across processes, so a
        # worker starts empty, with nobody listening and its logging left alone
        state = self.__dict__.copy()
        for key in ("_lock", "_local", "_active", "_logs", "_listeners", "_throughputs", "_handler"):
            state.pop(key, None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._reset()

    ### LISTENERS

    def subscribe(self, listener: Listener) -> None:
        """
        Call ``listener`` with every task and log event from now on.

        Listeners run on whichever thread reported the event and should return
        quickly. A listener that raises is logged and unsubscribed.
        """
        with self._lock:
            if listener not in self._listeners:
                self._listeners.append(listener)

    def unsubscribe(self, listener: Listener) -> None:
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)

    ### TASKS

    def task(
            self,
            name: str, *,
            total: float | None = None,
            detail: str = "",
            level: int = logging.INFO
    ) -> Task:
        """
        Open a task, best used as a context manager so it always finishes.

        Args:
            name (str): What is being done.
            total (float | None): Total units of work, None if unknown.
            detail (str): What the task is on right now, can be updated as it goes.
            level (int): Level the tasks completion is logged at, use DEBUG for
                routine tasks that would clutter the log.
        """
        with self._lock:
            stack = self._stack()
            task = Task(self, name, parent=stack[-1] if stack else None, total=total, detail=detail, level=level)
            self._active.append(task)

        logger.debug("%s started", name)
        self._emit(TaskEvent(EventKind.STARTED, task))
        return task

    def track(
            self,
            iterable: Iterable[T],
            name: str, *,
            total: float | None = None,
            detail: str = "",
            level: int = logging.INFO
    ) -> Iterator[T]:
        """Iterate while reporting each item as progress on a task, total from len() if not given."""
        if total is None and isinstance(iterable, Sized):
            total = len(iterable)

        with self.task(name, total=total, detail=detail, level=level) as task:
            for item in iterable:
                yield item
                task.advance()

    @property
    def active(self) -> tuple[Task, ...]:
        """Running tasks in the order they started."""
        with self._lock:
            return tuple(self._active)

    ### LOGS

    @property
    def logs(self) -> tuple[LogEntry, ...]:
        """Recent log entries, oldest first, up to the configured log history."""
        with self._lock:
            return tuple(self._logs)

    ### THROUGHPUT

    def throughput(self, name: str) -> Throughput:
        """Throughput of the batch stream ``name``, e.g. a split, created on first use."""
        with self._lock:
            throughput = self._throughputs.get(name)
            if throughput is None:
                throughput = self._throughputs[name] = Throughput(self, name)
            return throughput

    @property
    def throughputs(self) -> dict[str, Throughput]:
        with self._lock:
            return dict(self._throughputs)

    def close(self) -> None:
        if self._handler is not None:
            logging.getLogger(_PACKAGE_LOGGER).removeHandler(self._handler)
            self._handler = None

        with self._lock:
            self._listeners.clear()

    ### INTERNALS

    def _stack(self) -> list[Task]:
        """Tasks this thread is inside, innermost last."""
        stack = getattr(self._local, "stack", None)
        if stack is None:
            stack = self._local.stack = []
        return stack

    def _push(self, task: Task) -> None:
        stack = self._stack()
        stack.append(task)
        task._stack = stack

    def _pop(self, task: Task) -> None:
        # usually the innermost, but a generator can be closed after others were entered
        stack = task._stack
        if stack is not None and task in stack:
            stack.remove(task)

    def _updated(self, task: Task) -> None:
        if task.running:
            self._emit(TaskEvent(EventKind.UPDATED, task))

    def _end(self, task: Task, state: TaskState, error: BaseException | None = None) -> None:
        with self._lock:
            if not task.running:
                return

            task.ended = time.perf_counter()
            task.state = state
            task.error = error
            self._active.remove(task)

        progress = f" ({task.completed:g}/{task.total:g})" if task.total else ""

        if state is TaskState.DONE:
            logger.log(task.level, "%s done in %.2fs%s", task.name, task.elapsed, progress)
            self._emit(TaskEvent(EventKind.FINISHED, task))
        else:
            reason = f" ({type(error).__name__})" if error is not None else ""
            logger.warning("%s failed after %.2fs%s", task.name, task.elapsed, reason)
            self._emit(TaskEvent(EventKind.FAILED, task))

    def _log(self, record: logging.LogRecord) -> None:
        entry = LogEntry(time=record.created, level=record.levelno, logger=record.name, message=record.getMessage())

        with self._lock:
            self._logs.append(entry)

        self._emit(LogEvent(entry))

    def _emit(self, event: StatusEvent) -> None:
        # never called under the lock, a listener may take its own locks and read back from here
        for listener in tuple(self._listeners):
            try:
                listener(event)
            except Exception:
                logger.exception("status listener %r raised and was unsubscribed", listener)
                self.unsubscribe(listener)
