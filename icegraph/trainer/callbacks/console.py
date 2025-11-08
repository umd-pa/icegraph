# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from datetime import datetime
from typing import Optional, TYPE_CHECKING, List, Dict, Tuple, Union, Any

from rich.console import Console as RichConsole, Group
from rich.text import Text
from rich.panel import Panel
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn, TimeElapsedColumn
from rich.live import Live
from rich.layout import Layout
from rich.align import Align

from .callback import Callback
from icegraph.console import Console
from icegraph.types import ComputedMetrics

__all__ = ["ConsoleCallback"]

if TYPE_CHECKING:
    from .. import Trainer
else:
    class Trainer:
        pass


class ConsoleCallback(Callback):
    """Rich-based console UI for training, defaults to standard printouts on incompatible terminals (like IDE's)."""

    @staticmethod
    def _top_left(renderable) -> Align:
        return Align(renderable, align="left", vertical="top")

    @staticmethod
    def _panel(title: Optional[str], renderable, **kwargs) -> Panel:
        """Create a panel with top-left aligned content."""
        return Panel(ConsoleCallback._top_left(renderable), title=title, **kwargs)

    class LogPanel:
        def __init__(self, max_lines: int = 2000):
            self.lines: List[Text] = []
            self.max_lines = max_lines

        def write(self, text: Text) -> None:
            self.lines.append(text)
            self.lines = self.lines[-self.max_lines:]

        def render(self, display_lines: int) -> Panel:
            lines = Group(*self.lines[-display_lines:])
            return ConsoleCallback._panel("Logs", lines)

    def __init__(self, *, log_max_lines: int = 2000, **_) -> None:
        super().__init__()

        self.console = RichConsole()
        self.is_terminal = self.console.is_terminal

        # Rich state
        self.progress: Optional[Progress] = None
        self.task_id: Optional[int] = None
        self.live: Optional[Live] = None
        self.layout: Optional[Layout] = None
        self.logpanel: Optional[ConsoleCallback.LogPanel] = None

        # metrics snapshot
        self._latest_metrics: List[Dict[str, Any]] = []

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
            self.logpanel = self.LogPanel(max_lines=log_max_lines)

        # for metrics trends
        self.k: int = 5
        self._ema_alpha: float = 2.0 / (self.k + 1.0)

    def _tee_log(self, msg: str) -> None:
        self.console.log(msg)
        if self.logpanel is not None:
            self.logpanel.write(Text.assemble(
                (f"[{datetime.now().strftime('%H:%M:%S')}]  ", "white"),
                (msg, "grey")
            ))
            self._refresh_live()

    def _build_layout(self, trainer: "Trainer") -> Layout:
        layout = Layout()

        # Build header once
        title = Text("ICEGRAPH TRAINER", style="bold cyan")
        lines = [title, "", f"Output directory: {trainer.outdir}"]
        body = Group(*lines)

        _init_header_size = len(lines) + 4
        _init_top_size = 5
        _init_mid_size = 7

        layout.split_column(
            Layout(self._panel(None, body, padding=(1, 2)), name="header", size=_init_header_size),
            Layout(Panel(Align(self.progress, align="center"), padding=(1, 2), title="Progress"), name="top", size=_init_top_size),
            Layout(name="mid", size=_init_mid_size),
            Layout(self.logpanel.render(8), name="bottom", ratio=1, minimum_size=8),
        )

        layout["mid"].split_row(
            Layout(self._render_metrics(), name="stats", ratio=1),
            Layout(self._render_info(), name="info", ratio=2),
        )
        return layout

    def _render_metrics(self) -> Panel:
        lines = []
        if self._latest_metrics:
            for k, v in self._latest_metrics[-1].items():
                label = k.split(":")[-1].upper()
                try:
                    s = f"{v:.6g}" if isinstance(v, (int, float)) else str(v)
                except Exception:
                    s = str(v)
                lines.append(f"{label.ljust(10)}: {s}")
        if not lines:
            lines = ["—"]

        if self.layout and self.layout["mid"].size < (min_height := len(lines) + 4):
            self.layout["mid"].size = min_height

        return Panel(Align(Group(*lines), align="center"), title="Latest Val Metrics", padding=(1, 2))

    def _render_info(self) -> Panel:
        """Render the info panel for metrics trends."""
        lines: List[str] = []

        if self._latest_metrics:
            for key in self._latest_metrics[-1].keys():
                series = self._series_for_key(key)
                if not series:
                    continue
                ema = self._ema_window(series)
                delta = series[-1] - series[0]
                label = key.split(":")[-1].upper()
                lines.append(label.ljust(10) + "[EMA5:" + f"{ema:.4g}".rjust(12) + "  Δ5:" + f"{delta:.4g}".rjust(12) + "]")

        if not lines:
            lines = ["—"]

        if self.layout and self.layout["mid"].size < (min_height := len(lines) + 4):
            self.layout["mid"].size = min_height

        return Panel(Align(Group(*lines), align="center"), title="Metrics Trends", padding=(1, 2))

    def _refresh_live(self) -> None:
        if self.live and self.layout and self.logpanel:
            log_space_available = self.console.size.height - sum(
                c.size or 0 for c in self.layout.children if c.name != "bottom"
            ) - 3
            self.layout["bottom"].update(self.logpanel.render(log_space_available))

    def reset_progress_bar(self, desc: str, total: int) -> None:
        if self.task_id is None:
            self.task_id = self.progress.add_task(desc, total=total)
        else:
            self.progress.reset(self.task_id, total=total)
            self.progress.update(self.task_id, description=desc)

    def _series_for_key(self, key: str) -> List[float]:
        """Extract numeric sequence for `key` from the rolling window."""
        seq: List[float] = []
        for rec in self._latest_metrics:
            try:
                seq.append(float(rec[key]))
            except Exception:
                # Skip missing or non-numeric entries
                continue
        return seq

    def _ema_window(self, seq: List[float]) -> float:
        """EMA over the given window."""
        if not seq:
            return float("nan")
        ema = seq[0]
        a = self._ema_alpha
        for x in seq[1:]:
            ema = (1.0 - a) * ema + a * x
        return ema

    # ----------------------------- callbacks ----------------------------------
    def on_execute(self, trainer: Trainer) -> None:
        if not self.is_terminal:
            Console.out(f"Trainer output directory: {trainer.outdir}")
            if trainer.device.type == "cpu":
                Console.out("No accelerators found, falling back to CPU training.", severity=2)
            return

        self.layout = self._build_layout(trainer)
        self.live = Live(self.layout, console=self.console, refresh_per_second=10)
        self.live.start()

    def on_train_begin(self, trainer: Trainer, epoch: int) -> None:
        if not self.is_terminal:
            Console.out(f"[Train] Epoch {epoch + 1}/{trainer.trainer_config.max_epochs}")
            Console.out(f"Current LR --> {[pg['lr'] for pg in trainer.optimizer.param_groups]}")
            return

        desc = f"Train Epoch {epoch + 1}/{trainer.trainer_config.max_epochs}"
        total = len(trainer.registry.train_dataloader)

        self.reset_progress_bar(desc, total)

        self._tee_log(
            f"Starting epoch {epoch + 1}    LR={ [pg['lr'] for pg in trainer.optimizer.param_groups] }"
        )

    def on_batch_end(self, trainer, batch, out, target, loss, metrics) -> None:
        if self.is_terminal and self.task_id is not None and self.progress is not None:
            self.progress.advance(self.task_id)

    def on_validation_begin(self, trainer: Trainer, epoch: int) -> None:
        if not self.is_terminal:
            self._log(f"[Validation] Epoch {epoch + 1}/{trainer.trainer_config.max_epochs}")
            return

        desc = f"  Val Epoch {epoch + 1}/{trainer.trainer_config.max_epochs}"
        total = len(trainer.registry.val_dataloader)

        self.reset_progress_bar(desc, total)

    def on_test_begin(self, trainer: Trainer, epoch: int) -> None:
        if not self.is_terminal:
            self._log(f"[Test] Epoch {epoch + 1}/{trainer.trainer_config.max_epochs}")
            return

        desc = f" Test Epoch {epoch + 1}/{trainer.trainer_config.max_epochs}"
        total = len(trainer.registry.test_dataloader)

        self.reset_progress_bar(desc, total)

    def _display_metrics(self, metrics: ComputedMetrics, dataset: str) -> None:
        display = [f"{k.upper().split(':')[-1]}: {v:.4f}" if isinstance(v, (int, float)) else f"{k}: {v}"
                   for k, v in metrics.items()]
        self._log(f"[{dataset}] --> {' | '.join(display)}")

    def on_train_end(self, trainer: Trainer, epoch: int, metrics: ComputedMetrics) -> None:
        self._display_metrics(metrics, "train")

    # unify end hooks
    def on_validation_end(self, trainer: Trainer, epoch: int, metrics: ComputedMetrics) -> None:
        self._latest_metrics.append(metrics)
        self._latest_metrics = self._latest_metrics[-self.k:]

        if self.is_terminal and self.layout is not None:
            self.layout["stats"].update(self._render_metrics())
            self.layout["info"].update(self._render_info())

        self._display_metrics(metrics, "validation")

    def on_test_end(self, trainer: Trainer, epoch: int, metrics: ComputedMetrics) -> None:
        self._display_metrics(metrics, "test")

    def _log(self, text: str) -> None:
        if self.is_terminal:
            self._tee_log(text)
        else:
            Console.out(text)

    def on_teardown(self, trainer) -> None:
        if self.is_terminal:
            try:
                if self.live:
                    self.live.stop()
                if self.progress:
                    self.progress.stop()
            finally:
                self.live = None
                self.progress = None
                self.task_id = None
                self.layout = None
                self.logpanel = None

    # ENTRY POINTS

    def log(self, text: str) -> None:
        if self.is_terminal:
            self._tee_log(text)
        else:
            Console.out(text)
