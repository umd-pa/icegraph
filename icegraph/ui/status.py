# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

import time
from datetime import timedelta
from typing import TYPE_CHECKING

from rich.console import RenderableType
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from icegraph.engine.status import Status

__all__ = ["StatusView", "LogView"]


_SPINNER: str = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


class StatusView:
    """
    A status's running tasks, each indented under its parent with a spinner, a bar
    and count once its total is known, and its elapsed time.

    Reads the status when rendered, so inside a Live display it stays current
    without handling any events.
    """

    def __init__(self, status: Status, *, max_rows: int | None = None, bar_width: int = 30) -> None:
        self._status = status
        self._max_rows = max_rows
        self._bar_width = bar_width

    def __rich__(self) -> RenderableType:
        table = Table.grid(padding=(0, 1), expand=True)
        table.add_column(no_wrap=True, style="progress.spinner")
        table.add_column(ratio=1, no_wrap=True, overflow="ellipsis")
        table.add_column(no_wrap=True, width=self._bar_width)
        table.add_column(justify="right", no_wrap=True, style="progress.download")
        table.add_column(justify="right", no_wrap=True, style="progress.elapsed")

        for task in self._status.active[:self._max_rows]:
            label = Text("  " * task.depth + task.name)
            if task.detail:
                label.append(f"  {task.detail}", style="dim")

            bar: RenderableType = ""
            count = ""
            if task.total:
                bar = ProgressBar(total=task.total, completed=task.completed, width=self._bar_width)
                count = f"{task.completed:g}/{task.total:g}"

            table.add_row(
                _SPINNER[int(task.elapsed * 12) % len(_SPINNER)],
                label,
                bar,
                count,
                str(timedelta(seconds=int(task.elapsed)))
            )

        return table


class LogView:
    """
    A status's most recent log entries, newest last.

    Reads the status when rendered, so inside a Live display it stays current
    without handling any events.
    """

    def __init__(self, status: Status, *, rows: int = 5) -> None:
        self._status = status
        self._rows = rows

    def __rich__(self) -> RenderableType:
        entries = self._status.logs[-self._rows:] if self._rows > 0 else ()

        if not entries:
            return Text("-- nothing logged yet --", style="dim")

        table = Table.grid(padding=(0, 1), expand=True)
        table.add_column(no_wrap=True, style="log.time")
        table.add_column(no_wrap=True, width=8)
        table.add_column(ratio=1, no_wrap=True, overflow="ellipsis")

        for entry in entries:
            level = entry.levelname
            table.add_row(
                time.strftime("%H:%M:%S", time.localtime(entry.time)),
                Text(level, style=f"logging.level.{level.lower()}"),
                Text(entry.message)
            )

        return table
