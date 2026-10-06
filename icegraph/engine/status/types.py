# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Callable, TypeAlias

if TYPE_CHECKING:
    from .task import Task

__all__ = [
    "TaskState", "EventKind", "TaskEvent", "LogEntry", "LogEvent", "StatusEvent", "Listener", "ThroughputStats"
]


class TaskState(Enum):
    RUNNING = "running"
    DONE    = "done"
    FAILED  = "failed"


class EventKind(Enum):
    STARTED     = "started"
    UPDATED     = "updated"
    FINISHED    = "finished"
    FAILED      = "failed"


@dataclass(frozen=True, slots=True)
class TaskEvent:
    kind: EventKind

    # the live task, so fields reflect its state when read rather than when the event was made
    task: Task


@dataclass(frozen=True, slots=True)
class LogEntry:
    time:       float   # seconds since the epoch, as logging records it
    level:      int
    logger:     str
    message:    str

    @property
    def levelname(self) -> str:
        return logging.getLevelName(self.level)


@dataclass(frozen=True, slots=True)
class LogEvent:
    entry: LogEntry


StatusEvent: TypeAlias = TaskEvent | LogEvent

Listener: TypeAlias = Callable[[StatusEvent], None]


@dataclass(frozen=True, slots=True)
class ThroughputStats:
    # per second, over each meters window
    samples:    float
    batches:    float
    flops:      float   # 0 if not counted

    # fraction of time spent blocked
    waiting:    float
