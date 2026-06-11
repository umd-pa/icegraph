# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar
from functools import wraps

from rich.console import Group
from rich.text import Text
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn, TimeElapsedColumn, TaskID
from rich.live import Live
from rich.layout import Layout
from rich.align import Align

from torch import Tensor

from icegraph.common.data import Split
from icegraph.ui import console

from ..callback import Callback

if TYPE_CHECKING:
    from icegraph.trainer import Trainer
    from icegraph.engine.services.metrics import ComputedMetric

    from .. import context

__all__ = ["ConsoleCallback"]

# module logger
import logging
logger = logging.getLogger(__name__)


def terminal_only(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        if not getattr(self, "is_terminal", False):
            return None
        return fn(self, *args, **kwargs)
    return wrapper


class ConsoleCallback(Callback):
    """Rich-based console UI for training, defaults to standard printouts on incompatible terminals (like IDE's)."""

    # default dead-zone for the trend indicator
    _DEFAULT_EPS: ClassVar[float] = 1e-4

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

        # init mid panel: single combined metrics + trends table
        mid_panel = Layout(self._render_metrics(), name="mid", ratio=1, minimum_size=_init_mid_size)

        # stack all vertically
        layout.split_column(header_panel, progress_panel, mid_panel)

        # return initialized layout
        return layout

    @staticmethod
    def _as_list(x: Tensor | None) -> list[float] | None:
        if x is None:
            return None
        return x.reshape(-1).tolist()

    @classmethod
    def _cell(
        cls,
        value: float,
        ema: float | None,
        optimum: float | None
    ) -> Text:
        """
        One head's value plus a trend glyph.

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
        glyph  = "▲" if rising else "▼"  # dont feel like finding the ascii codes for these chars

        # desirability: did the gap to the optimum shrink vs the trend
        gap_now = abs(value - optimum)
        gap_ema = abs(ema   - optimum)
        diff    = gap_ema - gap_now          # > 0 means moved closer to optimum

        if abs(diff) < cls._DEFAULT_EPS:
            val.append(" –", style="dim")    # flat or jsut noise
            return val

        style = "bold green" if diff > 0 else "bold red"
        val.append(f" {glyph}", style=style)
        return val

    def _render_metrics(self) -> Panel:
        title = "Metrics/Trends"

        if not self._latest_metrics:
            placeholder = Align(Text("— no metrics yet —", style="dim"), align="center")
            return Panel(placeholder, title=title, padding=(1, 2))

        # flatten every metric's per-head fields up front; heads may differ in count
        prepared: list[tuple[ComputedMetric, list[float], list[float] | None, float | None]] = []
        max_heads = 0
        for metric in self._latest_metrics:
            values  = self._as_list(metric.value) or []
            emas    = self._as_list(metric.ema)
            optimum = metric.optimum
            max_heads = max(max_heads, len(values))
            prepared.append((metric, values, emas, optimum))

        table = Table(expand=True, header_style="bold", pad_edge=False, show_edge=False, border_style="dim")
        table.add_column("Metric", style="cyan", no_wrap=True)
        for h in range(max_heads):
            table.add_column(f"Head {h}", justify="right")

        for metric, values, emas, optimum in prepared:
            cells: list[Any] = [f"{metric.repr}  (s={metric.span})"]
            for h in range(max_heads):
                if h >= len(values):
                    cells.append("")                 # this metric has fewer heads
                    continue
                ema_h = emas[h] if (emas is not None and h < len(emas)) else None
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
                description=desc,
            )

    def _on_split_begin(self, trainer: Trainer, split: Split, total: int) -> None:
        epoch = trainer.current_epoch

        # format description
        desc = f"{split.name:>5} Epoch {epoch + 1}/{trainer.config.max_epochs}"

        # reset the progress bar for the start of the next split/epoch
        self.reset_progress_bar(desc, total)

    # callback hooks
    def on_execute(self, ctx: context.ExecuteContext) -> None:
        # only run on main rank
        if not ctx.trainer.state.is_main_process():
            self.is_terminal = False

        # no op if not in terminal
        if not self.is_terminal:
            return

        self.layout = self._build_layout(ctx.trainer)
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
        trainer = ctx.trainer
        self._on_split_begin(
            trainer, Split.TRAIN, len(trainer.data.dataloader(Split.TRAIN))
        )

    @terminal_only
    def on_validation_begin(self, ctx: context.ValidationBeginContext) -> None:
        trainer = ctx.trainer
        self._on_split_begin(
            trainer, Split.VAL, len(trainer.data.dataloader(Split.VAL))
        )

    @terminal_only
    def on_test_begin(self, ctx: context.TestBeginContext) -> None:
        trainer = ctx.trainer
        self._on_split_begin(
            trainer, Split.TEST, len(trainer.data.dataloader(Split.TEST))
        )

    @terminal_only
    def on_validation_end(self, ctx: context.ValidationEndContext) -> None:
        # no op if no layout has been initialized
        if self.layout is None:
            return

        # layout cannot be None past check
        self.layout: Layout

        metrics = ctx.trainer.metrics.compute()

        # cache validation metrics
        self._latest_metrics = metrics

        # refresh the combined metrics + trends panel
        self.layout["mid"].update(self._render_metrics())

    @terminal_only
    def on_teardown(self, ctx: context.TeardownContext) -> None:
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