# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar
from functools import wraps
import os

from rich.console import Group
from rich.text import Text
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn, TimeElapsedColumn, TaskID
from rich.live import Live
from rich.layout import Layout
from rich.align import Align

from torch import Tensor
import numpy as np

from icegraph.common.data import Split
from icegraph.engine.status import EventKind, TaskEvent
from icegraph.ui import console, StatusView, LogView

from ..callback import TrainerCallback

if TYPE_CHECKING:
    from icegraph.trainer import Trainer
    from icegraph.engine.services.metrics import MetricValue

    from .. import context

__all__ = ["ConsoleCallback"]

# module logger
import logging
logger = logging.getLogger(__name__)


def _si(value: float) -> str:
    """Scale to an SI prefix, e.g. 41200 -> '41.2k'."""
    for prefix in ("", "k", "M", "G", "T", "P"):
        if abs(value) < 1000:
            return f"{value:.3g}{prefix}"
        value /= 1000
    return f"{value:.3g}E"


class _Throughput:
    """The status's throughput for the split being run, formatted when rendered."""

    def __init__(self, trainer: Trainer) -> None:
        self._trainer = trainer

    def __rich__(self) -> Text:
        throughput = self._trainer.status.throughputs.get(self._trainer.split.value)
        if throughput is None:
            return Text("-- no throughput yet --", style="dim")

        current = throughput.stats()
        stats: list[tuple[str, str]] = []

        if current.samples:
            stats.append((_si(current.samples), "samples/s"))
            stats.append((f"{current.batches:.3g}", "batches/s"))

            if current.flops:
                stats.append((_si(current.flops), "FLOP/s"))

            stats.append((f"{current.waiting:.0%}", "waiting on data"))

        if not stats:
            return Text("-- no throughput yet --", style="dim")

        line = Text()
        for value, label in stats:
            if line:
                line.append("   ")
            line.append(value, style="bold")
            line.append(f" {label}", style="dim")

        return line


def terminal_only(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        if not getattr(self, "is_terminal", False):
            return None
        return fn(self, *args, **kwargs)
    return wrapper


class ConsoleCallback(TrainerCallback):
    """Rich-based console UI for training, defaults to standard printouts on incompatible terminals (like IDE's)."""

    # default dead-zone for the trend indicator
    _DEFAULT_EPS: ClassVar[float] = 1e-4

    # running tasks shown in the progress panel
    _ACTIVITY_ROWS: ClassVar[int] = 3

    # recent log entries shown in the log panel
    _LOG_ROWS: ClassVar[int] = 5

    def __init__(self) -> None:
        super().__init__()

        self.console = console

        # is terminal will be flipped to false on non-main rank
        self.is_terminal = self.console.is_terminal

        # rich state
        self.task_id:   TaskID      | None = None
        self.live:      Live        | None = None
        self.layout:    Layout      | None = None
        self.progress:  Progress    | None = None

        # running backend tasks shown on their own during startup
        self.startup:   Live        | None = None
        self._closed:   bool               = False

        # metrics snapshot
        self._latest_metrics: list[MetricValue] = []

        # init progress bar and logs only if in terminal
        if self.is_terminal:
            self.progress = Progress(
                TextColumn("{task.description}"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                TimeElapsedColumn(),
                TextColumn("| ETA:"),
                TimeRemainingColumn(),
                transient=False,
                expand=True,
            )

    @staticmethod
    def _top_left(renderable) -> Align:
        return Align(renderable, align="left", vertical="top")

    def _panel(self, title: str | None, renderable, **kwargs) -> Panel:
        """Create a panel with top-left aligned content."""
        return Panel(self._top_left(renderable), title=title, **kwargs)

    def _build_layout(self, trainer: Trainer) -> Layout:
        # init empty layout
        layout = Layout()

        # section sizes
        _init_header_size   = 7  #  (3 lines + 2 padding + 2 border)
        _init_top_size      = 6 + self._ACTIVITY_ROWS  #  (progress + throughput + running tasks + 2 padding + 2 border)
        _init_mid_size      = 7  #  may be resized for > 3 metrics, this is a minimum starting value

        # init header panel
        title = Text("ICEGRAPH TRAINER", style="bold cyan")
        lines = [title, "", f"Output directory: {trainer.outdir!s}"]
        body = Group(*lines)

        header_panel = Layout(self._panel(None, body, padding=(1, 2)), name="header", size=_init_header_size)

        # init progress panel
        progress = self.progress if self.progress is not None else Text("-- no progress to show --")
        activity = StatusView(trainer.status, max_rows=self._ACTIVITY_ROWS)
        progress_panel = Layout(
            Panel(
                Group(progress, _Throughput(trainer), activity), padding=(1, 2), title="Progress"
            ), name="top", size=_init_top_size
        )

        # init mid panel: single combined metrics + trends table
        mid_panel = Layout(self._render_metrics(), name="mid", ratio=1, minimum_size=_init_mid_size)

        # init log panel
        log_panel = Layout(
            Panel(LogView(trainer.status, rows=self._LOG_ROWS), padding=(0, 2), title="Log"),
            name="logs", size=self._LOG_ROWS + 2
        )

        # stack all vertically
        layout.split_column(header_panel, progress_panel, mid_panel, log_panel)

        # return initialized layout
        return layout

    @staticmethod
    def _as_array(values: tuple[Tensor | None, ...]) -> np.ndarray:
        """Stack per-head values into [L, W]; missing entries are nan."""
        width = max((0 if v is None else v.numel() for v in values), default=0)

        out = np.full((len(values), width), np.nan)

        for h, v in enumerate(values):
            if v is not None:
                out[h, :v.numel()] = v.reshape(-1).numpy()

        return out

    @classmethod
    def _entry(
            cls,
            value: float,
            ema: float | None,
            optimum: float | None
    ) -> Text:
        """
        One value plus a trend glyph.

        Arrow shows raw direction relative to the smoothed trend (value vs ema).
        Color shows desirability: green if this reading is closer to the
        metric's optimum than its ema (gap shrank), red if farther (gap grew),
        dim '-' if flat.
        """
        val = Text(f"{value:.4g}")

        # no smoothing history yet so direction undefined, show value only
        if ema is None:
            return val

        rising = value > ema
        glyph = "▲" if rising else "▼"  # dont feel like finding the ascii codes for these chars

        # desirability: did the gap to the optimum shrink vs the trend
        if optimum is not None and ema is not None:
            diff = abs(ema - optimum) - abs(value - optimum)  # > 0 means moved closer to optimum
        else:
            diff = 0

        if abs(diff) < cls._DEFAULT_EPS:
            val.append(" –", style="dim")  # flat or jsut noise
            return val

        style = "bold green" if diff > 0 else "bold red"
        val.append(f" {glyph}", style=style)
        return val

    @classmethod
    def _cell(
            cls,
            values: np.ndarray,
            emas: np.ndarray | None,
            optimum: float | None
    ) -> Text:
        """One head's values, each with its own trend glyph."""
        cell = Text()

        if emas is not None and emas.size != 0 and emas.shape != values.shape:
            raise ValueError(
                f"Both values and emas array must be of same shape, got values: {values.shape}, emas: {emas.shape}."
            )

        for i, value in enumerate(values):
            if np.isnan(value):
                continue

            # add spacing between entries
            if len(cell) != 0:
                cell.append("  ")

            ema = None if (emas is None or np.isnan(emas[i])) else float(emas[i])
            cell.append_text(cls._entry(float(value), ema, optimum))

        return cell

    def _render_metrics(self) -> Panel:
        title = "Metrics/Trends"

        if not self._latest_metrics:
            placeholder = Align(Text("-- no metrics yet --", style="dim"), align="center")
            return Panel(placeholder, title=title, padding=(1, 2))

        # flatten every metric per head up front
        prepared: list[tuple[MetricValue, np.ndarray, np.ndarray, float | None]] = []
        max_heads = 0
        for metric in self._latest_metrics:
            values = self._as_array(metric.value)
            emas = self._as_array(metric.ema)
            max_heads = max(max_heads, values.shape[0])
            prepared.append((metric, values, emas, metric.optimum))

        table = Table(expand=True, header_style="bold", pad_edge=False, show_edge=False, border_style="dim")
        table.add_column("Metric", style="cyan", no_wrap=True)
        for h in range(max_heads):
            table.add_column(f"Head {h}", justify="right")

        for metric, values, emas, optimum in prepared:
            cells: list[Any] = [f"{metric.repr}  (s={metric.span})"]
            for h in range(max_heads):
                if h >= values.shape[0]:
                    cells.append("")
                    continue
                ema_h = emas[h] if h < emas.shape[0] else None
                cells.append(self._cell(values[h], ema_h, optimum))
            table.add_row(*cells)

        return Panel(table, title=title, padding=(1, 2))

    def reset_progress_bar(self, desc: str, total: int) -> None:
        if self.progress is None:
            return

        if self.task_id is None:
            self.task_id = self.progress.add_task(desc, total=total)
        else:
            self.progress.reset(
                self.task_id,
                total=total,
                description=desc
            )

    def _on_split_begin(self, trainer: Trainer, split: Split, total: int) -> None:
        epoch = trainer.current_epoch

        # format description
        desc = f"{split.name:>5} Epoch {epoch + 1}/{trainer.config.max_epochs}"

        # reset the progress bar for the start of the next split/epoch
        self.reset_progress_bar(desc, total)

    def _stop_startup(self) -> None:
        if self.startup is not None:
            self.startup.stop()
            self.startup = None

    # callback hooks
    def on_init(self, ctx: context.InitContext) -> None:
        # only the main rank draws, read from the launch environment since services may not exist yet
        if int(os.environ.get("RANK", 0)) != 0:
            self.is_terminal = False

    @terminal_only
    def on_status(self, ctx: context.StatusContext) -> None:
        # setup runs before on_execute, so its first task opens a display of its own until
        # training takes over, the views read the status themselves when rendering
        event = ctx.event
        if not isinstance(event, TaskEvent) or event.kind is not EventKind.STARTED:
            return

        if self.layout is None and self.startup is None and not self._closed:
            # transient, finished tasks stay behind as log lines
            self.startup = Live(StatusView(ctx.engine.status), console=self.console, refresh_per_second=10, transient=True)
            self.startup.start()

    def on_execute(self, ctx: context.ExecuteContext) -> None:
        # only run on main rank
        if not ctx.engine.state.is_main_process():
            self.is_terminal = False

        # no op if not in terminal
        if not self.is_terminal:
            return

        # hand over from the startup display
        self._stop_startup()

        self.layout = self._build_layout(ctx.engine)
        self.live = Live(
            self.layout,
            console=self.console,
            refresh_per_second=4,
            screen=True,
            redirect_stdout=True,
            redirect_stderr=True
        )
        self.live.start()

    @terminal_only
    def on_batch_end(self, ctx: context.BatchEndContext) -> None:
        # no op if no progress bar is defined
        if self.task_id is None or self.progress is None:
            return

        self.progress.advance(self.task_id)

    @terminal_only
    def on_train_begin(self, ctx: context.TrainBeginContext) -> None:
        trainer = ctx.engine
        self._on_split_begin(
            trainer, Split.TRAIN, len(trainer.get_dataloader(Split.TRAIN))
        )

    @terminal_only
    def on_validation_begin(self, ctx: context.ValidationBeginContext) -> None:
        trainer = ctx.engine
        self._on_split_begin(
            trainer, Split.VAL, len(trainer.get_dataloader(Split.VAL))
        )

    @terminal_only
    def on_test_begin(self, ctx: context.TestBeginContext) -> None:
        trainer = ctx.engine
        self._on_split_begin(
            trainer, Split.TEST, len(trainer.get_dataloader(Split.TEST))
        )

    @terminal_only
    def on_validation_end(self, ctx: context.ValidationEndContext) -> None:
        # no op if no layout has been initialized
        if self.layout is None:
            return

        metrics = ctx.engine.metrics.compute(Split.VAL)

        # cache validation metrics
        self._latest_metrics = metrics

        # refresh the combined metrics + trends panel
        self.layout["mid"].update(self._render_metrics())

    @terminal_only
    def on_teardown(self, ctx: context.TeardownContext) -> None:
        self._closed = True

        try:
            # stop live and progress bar
            self._stop_startup()
            if self.live:
                self.live.stop()
            if self.progress:
                self.progress.stop()
        finally:

            # reset instance
            self.live       = None
            self.progress   = None
            self.task_id    = None
            self.layout     = None