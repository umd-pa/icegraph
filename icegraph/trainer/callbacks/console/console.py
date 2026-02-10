# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from rich.console import Console as RichConsole, Group
from rich.text import Text
from rich.panel import Panel
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn, TimeElapsedColumn, TaskID
from rich.live import Live
from rich.layout import Layout
from rich.align import Align

from icegraph.types.data import Split

from ..callback import Callback

if TYPE_CHECKING:
    from icegraph.trainer import Trainer
    from .. import context
    from icegraph.trainer.services.metrics import ComputedMetric

__all__ = ["ConsoleCallback"]

# module logger
import logging
logger = logging.getLogger(__name__)


class ConsoleCallback(Callback):
    """Rich-based console UI for training, defaults to standard printouts on incompatible terminals (like IDE's)."""

    @staticmethod
    def _top_left(renderable) -> Align:
        return Align(renderable, align="left", vertical="top")

    def _panel(self, title: str | None, renderable, **kwargs) -> Panel:
        """Create a panel with top-left aligned content."""
        return Panel(self._top_left(renderable), title=title, **kwargs)

    def __init__(self) -> None:
        super().__init__()

        self.console = RichConsole()
        self.is_terminal = self.console.is_terminal

        # rich state
        self.task_id:   TaskID      | None = None
        self.live:      Live        | None = None
        self.layout:    Layout      | None = None
        self.progress:  Progress    | None = None

        # metrics snapshot
        self._latest_metrics: list[ComputedMetric] = []

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

    def _build_layout(self, trainer: Trainer) -> Layout:
        # init empty layout
        layout = Layout()

        # section sizes
        _init_header_size   = 7  #  (3 lines + 2 padding + 2 border)
        _init_top_size      = 5  #  (1 progress bar + 2 padding + 2 border)
        _init_mid_size      = 7  #  may be resized for > 3 metrics, this is a minimum starting value

        # init header panel
        title = Text("ICEGRAPH TRAINER", style="bold cyan")
        lines = [title, "", f"Output directory: {trainer.outdir!s}"]
        body = Group(*lines)

        header_panel = Layout(self._panel(None, body, padding=(1, 2)), name="header", size=_init_header_size)

        # init progres panel
        progress_panel = Layout(
            Panel(
                Align(self.progress, align="center"), padding=(1, 2), title="Progress"
            ), name="top", size=_init_top_size
        )

        # init mid panel (for metrics)
        mid_panel = Layout(name="mid", ratio=1, minimum_size=_init_mid_size)

        # stack all vertically
        layout.split_column(header_panel, progress_panel, mid_panel)

        # split mid for metrics and trends
        layout["mid"].split_row(
            Layout(self._render_metrics(), name="stats", ratio=2),
            Layout(self._render_info(), name="info", ratio=3)
        )

        # return initialized layout
        return layout

    def _render_metric_panel(self, title: str, value_fn: Callable[[ComputedMetric], str]) -> Panel:
        lines: list[str] = []

        for metric in self._latest_metrics:
            formatted_value = value_fn(metric)
            label = metric.name.upper().ljust(10)
            lines.append(f"{label}: {formatted_value}")

        if not lines:
            lines.append("—")

        content = Align(Group(*(Text(line) for line in lines)), align="center")
        return Panel(content, title=title, padding=(1, 2))

    def _render_metrics(self) -> Panel:
        """Render the metrics panel."""

        def value_fn(metric: ComputedMetric) -> str:
            return (
                f"{metric.value:.6g}" if isinstance(metric.value, (int, float)) else str(metric.value)
            )

        return self._render_metric_panel("Latest Val Metric", value_fn)

    def _render_info(self) -> Panel:
        """Render the info panel for metrics trends."""

        def value_fn(metric: ComputedMetric) -> str:
            return (
                f"[EMA{metric.span}:{metric.ema:>12.4g}  Δ{metric.span}:{metric.delta:>12.4g}]"
            )

        return self._render_metric_panel("Metric Trends", value_fn)

    def reset_progress_bar(self, desc: str, total: int) -> None:
        # no op if not in terminal
        if not self.is_terminal:
            return

        # ensure task exists
        if self.task_id is None:
            self.task_id = self.progress.add_task(desc, total=total)
        else:
            # reset and update the progress bar
            self.progress.reset(self.task_id, total=total, description=desc)

    # callback hooks
    def on_execute(self, trainer: Trainer) -> None:
        # no op if not in terminal
        if not self.is_terminal:
            return

        self.layout = self._build_layout(trainer)
        self.live   = Live(
            self.layout,
            console=self.console,
            refresh_per_second=10,
            screen=True,
            redirect_stdout=True,
            redirect_stderr=True
        )
        self.live.start()

    def on_batch_end(self, ctx: context.BatchEndContext) -> None:
        # no op if not in terminal
        if not self.is_terminal:
            return

        # no op if no progress bar is defined
        if self.task_id is None or self.progress is None:
            return

        self.progress.advance(self.task_id)

    def _on_split_begin(self, trainer: Trainer, split: Split, total: int) -> None:
        epoch = trainer.current_epoch

        # format description
        desc = f"{split.name:>5} Epoch {epoch + 1}/{trainer.config.trainer.max_epochs}"

        # reset the progress bar for the start of the next split/epoch
        self.reset_progress_bar(desc, total)

    def on_train_begin(self, ctx: context.TrainBeginContext) -> None:
        trainer = ctx.trainer
        self._on_split_begin(
            trainer, Split.TRAIN, len(trainer.data.dataloader(Split.TRAIN))
        )

    def on_validation_begin(self, ctx: context.ValidationBeginContext) -> None:
        trainer = ctx.trainer
        self._on_split_begin(
            trainer, Split.VAL, len(trainer.data.dataloader(Split.VAL))
        )

    def on_test_begin(self, ctx: context.TestBeginContext) -> None:
        trainer = ctx.trainer
        self._on_split_begin(
            trainer, Split.TEST, len(trainer.data.dataloader(Split.TEST))
        )

    def on_validation_end(self, ctx: context.ValidationEndContext) -> None:
        # no op if not in terminal
        if not self.is_terminal:
            return

        # no op if no layout has been initialized
        if self.layout is None:
            return

        metrics = ctx.trainer.metrics.compute()

        # cache validation metrics
        self._latest_metrics = metrics

        # update info and stats
        self.layout["stats"].update(self._render_metrics())
        self.layout["info"].update(self._render_info())

    def on_teardown(self, ctx: context.TeardownContext) -> None:
        # no op if not in terminal
        if not self.is_terminal:
            return

        try:
            # stop live and progress bar
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
