# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple, Self, TypeVar
from collections.abc import Iterable, Iterator
from pathlib import Path
import time

from rich.console import Group, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

from icegraph.ui import console

if TYPE_CHECKING:
    from multiprocessing.context import BaseContext

    from .shared.queue import IterableQueue
    from .envelope import Envelope

__all__ = ["Tally", "Step", "PipelineConsole"]


T = TypeVar("T")


def _duration(ms: float) -> str:
    """Milliseconds, in seconds once past one."""
    return f"{ms:.0f}ms" if ms < 1000 else f"{ms / 1000:.2f}s"


class Tally:
    """
    Counts each stage keeps in shared memory as items pass through it, for the console to draw: the items the
    stage holds right now and the items it has finished. Indexed by stage index.
    """

    def __init__(self, stages: int, ctx: BaseContext) -> None:
        self._lock = ctx.Lock()
        self.busy = ctx.RawArray("i", stages)
        self.done = ctx.RawArray("i", stages)

    def count(self, src: Iterable[T], index: int) -> Iterator[T]:
        """Pass on the items of `src` to stage `index`, which holds each one until it asks for the next."""
        for item in src:
            with self._lock:
                self.busy[index] += 1

            yield item

            with self._lock:
                self.busy[index] -= 1
                self.done[index] += 1


class Step(NamedTuple):
    """A stage as the console draws it. Steps are listed in pipeline order, so a step's position is its stage index."""
    group:   str                        # extract, collect, process or write
    name:    str                        # the stage's plugin name
    workers: int                        # processes running the stage
    queue:   IterableQueue[Any] | None  # the queue feeding the stage, None between processors chained in one process


class PipelineConsole:
    """
    Terminal display of a running pipeline. Shows overall progress, then every stage top to bottom with its busy
    workers, the items it has finished and its time per shard, and between stages the queues the data moves through.

    Reads the tally and the queues each time it is drawn so it stays current on its own.
    """

    def __init__(self, outdir: Path, total: int, steps: list[Step], tally: Tally) -> None:
        self.console = console

        self._outdir = outdir
        self._steps = steps
        self._tally = tally
        self._start = time.monotonic()

        # written so far
        self.files = 0
        self.shards = 0
        self.events = 0

        # running mean of each metric, per shard
        self.metrics: dict[str, float] = {}

        self.progress = Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            TextColumn("| ETA:"),
            TimeRemainingColumn(),
            expand=True,
            speed_estimate_period=300.0
        )
        self._task = self.progress.add_task("Files", total=total)

        self.live = Live(self, console=self.console, refresh_per_second=4)

    def __enter__(self) -> Self:
        self.live.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.live.stop()

    def add(self, item: Envelope) -> None:
        """Count a shard the writers finished."""
        # each shard holds every file collected into it
        count = len(item.get_local_attr("sources"))

        self.files += count
        self.shards += 1
        self.events += item.main.height

        # running mean of each metric
        for key, value in item.metrics.items():
            self.metrics[key] = self.metrics.get(key, 0.0) + (value - self.metrics.get(key, 0.0)) / self.shards

        self.progress.advance(self._task, count)

    def _totals(self) -> Text:
        elapsed = max(time.monotonic() - self._start, 1e-9)

        stats = [
            (f"{self.shards:,}", "shards"),
            (f"{self.events:,}", "events"),
            (f"{self.files / elapsed:.3g}", "files/s"),
            (f"{self.events / elapsed:,.0f}", "events/s")
        ]

        line = Text()
        for value, label in stats:
            if line:
                line.append("   ")
            line.append(value, style="bold")
            line.append(f" {label}", style="dim")

        return line

    @staticmethod
    def _workers(busy: int, workers: int) -> Text:
        return Text.assemble(("\N{BLACK CIRCLE}" * busy, "bold green"), ("\N{WHITE CIRCLE}" * (workers - busy), "dim"))

    @staticmethod
    def _queue(queue: IterableQueue[Any]) -> Text:
        # only the source files are fed unbounded, all at once
        if not queue.maxsize:
            return Text(f"{queue.qsize():,} waiting", style="dim")

        # a full queue holds back the stages upstream of it
        size = min(queue.qsize(), queue.maxsize)
        style = "bold yellow" if size == queue.maxsize else "cyan"

        return Text.assemble(
            ("\N{BLACK PARALLELOGRAM}" * size, style),
            ("\N{WHITE PARALLELOGRAM}" * (queue.maxsize - size), "dim")
        )

    def _flow(self) -> RenderableType:
        table = Table(box=None, pad_edge=False, header_style="dim")
        table.add_column()
        table.add_column(style="cyan", no_wrap=True)
        table.add_column(no_wrap=True)
        table.add_column("finished", justify="right", no_wrap=True)
        table.add_column("per shard", justify="right", no_wrap=True)

        last = None
        for index, step in enumerate(self._steps):
            if step.queue is not None:
                table.add_row("", Text("\N{BOX DRAWINGS LIGHT VERTICAL}", style="dim"), self._queue(step.queue))

            # extractors and the collector take whole files, every stage after takes shards
            unit = "files" if step.group in ("extract", "collect") else "shards"

            # the collector only groups files, so records no time
            ms = self.metrics.get(f"{index}_{step.name}")

            table.add_row(
                Text(step.group if step.group != last else "", style="bold"),
                step.name,
                self._workers(self._tally.busy[index], step.workers),
                f"{self._tally.done[index]:,} {unit}",
                "" if ms is None else _duration(ms)
            )
            last = step.group

        legend = Text(
            "\N{BLACK CIRCLE} busy   \N{WHITE CIRCLE} idle   "
            "\N{BLACK PARALLELOGRAM} queued   \N{WHITE PARALLELOGRAM} free",
            style="dim"
        )
        return Group(table, "", legend)

    def __rich__(self) -> RenderableType:
        title = Text("ICEGRAPH PIPELINE", style="bold cyan")
        header = Group(title, "", f"Output directory: {self._outdir!s}")

        return Group(
            Panel(header, padding=(1, 2)),
            Panel(Group(self.progress, self._totals()), title="Progress", padding=(1, 2)),
            Panel(self._flow(), title="Flow", padding=(1, 2))
        )
