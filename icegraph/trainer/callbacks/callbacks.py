# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from datetime import datetime
from typing import Optional, TYPE_CHECKING, List, Dict, Tuple, Union, Mapping, Any
from pathlib import Path

import torch
from rich.console import Console as RichConsole, Group
from rich.text import Text
from rich.panel import Panel
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn, TimeElapsedColumn
from rich.live import Live
from rich.layout import Layout
from rich.align import Align

from .base import Callback
from icegraph.console import Console
from icegraph.data.base import IGData
from icegraph.types import ComputedMetrics, MetricsPlotMethod
from icegraph.inference import CoreModel
from icegraph.trainer.tensorboard import TensorBoard
from icegraph.renderer import ParityPlot, BiasPlot, ConfusionMatrixPlot, ROCPlot
from icegraph._version import __version__

__all__ = [
    "TensorBoardCallback",
    "ExportCallback",
    "ConsoleCallback",
    "RegressionMetricsCallback",
    "MulticlassMetricsCallback"
]

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

    def __init__(
            self,
            *,
            log_max_lines: int = 2000,
    ) -> None:
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
    def on_train_begin(self, trainer: "Trainer") -> None:
        if not self.is_terminal:
            Console.out(f"Trainer output directory: {trainer.outdir}")
            if trainer.device.type == "cpu":
                Console.out("No accelerators found, falling back to CPU training.", severity=2)
            return

        self.layout = self._build_layout(trainer)
        self.live = Live(self.layout, console=self.console, refresh_per_second=4)
        self.live.start()

    def on_epoch_begin(self, trainer: "Trainer", epoch: int) -> None:
        if not self.is_terminal:
            Console.out(f"[Train] Epoch {epoch + 1}/{trainer.trainer_config.max_epochs}")
            Console.out(f"Current LR --> {[pg['lr'] for pg in trainer.optimizer.param_groups]}")
            return

        desc = f"Train Epoch {epoch + 1}/{trainer.trainer_config.max_epochs}"
        total = trainer.train_batch_count

        self.reset_progress_bar(desc, total)

        self._tee_log(
            f"Starting epoch {epoch + 1}    LR={ [pg['lr'] for pg in trainer.optimizer.param_groups] }"
        )

    def on_batch_end(self, trainer, batch, out, target, loss, metrics) -> None:
        if self.is_terminal and self.task_id is not None and self.progress is not None:
            self.progress.advance(self.task_id)

    def on_validation_begin(self, trainer: "Trainer", epoch: int) -> None:
        if not self.is_terminal:
            self._log(f"[Validation] Epoch {epoch + 1}/{trainer.trainer_config.max_epochs}")
            return

        desc = f"  Val Epoch {epoch + 1}/{trainer.trainer_config.max_epochs}"
        total = trainer.train_batch_count

        self.reset_progress_bar(desc, total)

    def on_test_begin(self, trainer: "Trainer", epoch: int) -> None:
        if not self.is_terminal:
            self._log(f"[Test] Epoch {epoch + 1}/{trainer.trainer_config.max_epochs}")
            return

        desc = f" Test Epoch {epoch + 1}/{trainer.trainer_config.max_epochs}"
        total = trainer.train_batch_count

        self.reset_progress_bar(desc, total)

    def _display_metrics(self, metrics: ComputedMetrics, dataset: str) -> None:
        display = [f"{k.upper().split(':')[-1]}: {v:.4f}" if isinstance(v, (int, float)) else f"{k}: {v}"
                   for k, v in metrics.items()]
        self._log(f"[{dataset}] --> {' | '.join(display)}")

    def on_epoch_end(self, trainer: "Trainer", epoch: int, metrics: ComputedMetrics) -> None:
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


class TensorBoardCallback(Callback):

    def __init__(self) -> None:
        self._tb: Optional[TensorBoard] = None

    def on_init(self, trainer) -> None:
        if self._tb is None:
            self._tb = TensorBoard(trainer.log_dir)
        pid, port = self._tb.launch()

        trainer.console.log(f"TensorBoard started with PID {pid} at http://localhost:{port}")

    def on_epoch_end(self, trainer, epoch, metrics) -> None:
        for metric, value in metrics.items():
            self._tb.writer.add_scalar(f"Train/{metric.upper().split(':')[-1]}", value, epoch + 1)

    def on_validation_end(self, trainer, epoch, metrics) -> None:
        for metric, value in metrics.items():
            self._tb.writer.add_scalar(f"Validation/{metric.upper().split(':')[-1]}", value, epoch + 1)

    def on_test_end(self, trainer, epoch, metrics) -> None:
        for metric, value in metrics.items():
            self._tb.writer.add_scalar(f"Test/{metric.upper().split(':')[-1]}", value, epoch + 1)

    def on_teardown(self, trainer) -> None:
        self._tb.writer.close()
        self._tb.shutdown()


class ExportCallback(Callback):
    def __init__(self) -> None:
        self._best_loss: float = float("inf")

    @staticmethod
    def _generate_model(trainer: Trainer) -> CoreModel:
        # TODO: only pass necessary attrs, IGData.attrs can be very large and needs to be stripped
        export_model = CoreModel(
            net=trainer.model,
            normalizer=trainer.normalizer,
            metadata={
                **IGData.attrs,
                "model": {
                    "version": __version__,
                    "timestamp": datetime.now().timestamp()
                }
            }
        )
        return export_model

    def _export(self, trainer: Trainer, epoch: int, metrics: ComputedMetrics) -> None:
        models_dir = trainer.outdir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        latest_path = models_dir / "model_latest.pt"
        best_path = models_dir / "model_best.pt"

        # Build CoreModel for export
        export_model = self._generate_model(trainer)

        trainer.console.log(f"Saving latest model to {latest_path}")
        try:
            torch.save(export_model, latest_path)
        except Exception as e:
            trainer.console.log(f"Failed to save model: {e}")

        # Save best model if improved
        if metrics is not None:
            loss_key = next((k for k in metrics if k.startswith("loss")), None)
            loss = metrics[loss_key]
            if loss < self._best_loss:
                trainer.console.log(
                    f"New best {loss_key.split(':')[1].upper()} {loss:.4f} < {self._best_loss:.4f}; "
                    f"saving model to {best_path}"
                )
                self._best_loss = loss
                try:
                    torch.save(export_model, best_path)
                except Exception as e:
                    trainer.console.log(f"Failed to save model: {e}")

    # run both on validation and test, not on train
    on_validation_end = on_test_end = _export

    def on_epoch_end(self, trainer, epoch, metrics) -> None:
        # if current interval is a save interval, save a persistent copy
        if (epoch + 1) % trainer.trainer_config.save_interval == 0:
            export_model = self._generate_model(trainer)
            persistent_path = trainer.outdir / "models" / f"model.epoch_{epoch + 1}.pt"
            trainer.console.log(f"Saving persistent model to {persistent_path}")
            try:
                torch.save(export_model, persistent_path)
            except Exception as e:
                trainer.console.log(f"Failed to save model: {e}", severity=3)


class PlotConfigurationMixin:

    _ALIASES:   Dict[str, List[str]]
    _plotters:  List[Tuple[MetricsPlotMethod, Dict]]
    _dispatch:  Dict[str, MetricsPlotMethod]

    def _parse_options(self, options: Union[Dict[str, Dict], List[str]]) -> Dict[str, Dict]:
        # build lookup dict
        alias_lookup = {
            alias: canonical
            for canonical, aliases in self._ALIASES.items()
            for alias in aliases
        }

        # normalize dict keys in place
        if isinstance(options, list):
            options = {o: {} for o in options}
        for k, v in list(options.items()):
            norm_key = alias_lookup.get(k, k)
            if norm_key != k:
                options[norm_key] = options.pop(k)

        return options

    def configure_plots(self, options: Union[Dict[str, Dict], List[str]]) -> None:
        options = self._parse_options(options)

        for option, kwargs in options.items():
            if option not in self._dispatch.keys():
                raise KeyError(f"Option '{option}' is not supported.")
            self._plotters.append((self._dispatch[option], kwargs))

    @staticmethod
    def _plot_dir(trainer: Trainer) -> Path:
        outdir: Path = trainer.outdir / "plots"
        outdir.mkdir(parents=True, exist_ok=True)
        return outdir


class RegressionMetricsCallback(Callback, PlotConfigurationMixin):

    _COMPATIBLE = ["regression"]
    _ALIASES: Dict[str, List[str]] = {}

    def __init__(self) -> None:
        self._y_asinh_mask:     Optional[List[str]] = None
        self._target_labels:    Optional[List[str]] = None
        self._include_labels:   Optional[List[str]] = None

        # plotting attrs
        self._plotters: List[Tuple[MetricsPlotMethod, Dict]] = []

        # dispatch dict
        self._dispatch: Dict[str, MetricsPlotMethod] = {
            "bias": self._build_bias_plot,
            "parity": self._build_parity_plot
        }

    def on_init(self, trainer: Trainer) -> None:
        self._y_asinh_mask =    IGData.attrs[0]["global"]["apply_log_scaling_y"]
        self._include_labels =  IGData.attrs[0]["global"]["include_labels"]
        self._target_labels =   IGData.attrs[0]["global"]["target_labels"]

    def on_test_end(self, trainer: Trainer, epoch: int, metrics: ComputedMetrics) -> None:
        for plotter, kwargs in self._plotters:
            plotter(trainer, epoch, "test", **kwargs)

    def on_validation_end(self, trainer: Trainer, epoch: int, metrics: ComputedMetrics) -> None:
        for plotter, kwargs in self._plotters:
            plotter(trainer, epoch, "val", **kwargs)

    def _build_parity_plot(self, trainer: Trainer, epoch: int, dataset: str, **kwargs) -> None:
        preds = trainer.last_eval[dataset]["preds"]
        targs = trainer.last_eval[dataset]["targets"]

        n_cols = preds.shape[1]

        for i in range(n_cols):
            label = self._target_labels[i]

            pred = preds[:, i]
            targ = targs[:, i]

            axis_title = r"\text{%s}" % label

            if self._target_labels[i] in self._y_asinh_mask:
                pred = torch.log10(pred)
                targ = torch.log10(targ)

                axis_title = r"log_{10}\left[%s\right]" % axis_title

            layout_kwargs = {
                "title": f"{label} Parity [Epoch {epoch + 1} - {dataset.title()}]",
                "yaxis_title": r"$\text{Predicted }%s$" % axis_title,
                "xaxis_title": r"$\text{True }%s$" % axis_title
            }

            save_path = self._plot_dir(trainer) / f"{label}.parity.{epoch + 1}.html"

            plot = ParityPlot()
            plot.plot(
                x=targ,
                y=pred,
                save_path=save_path,
                layout_kwargs=layout_kwargs
            )
            trainer.console.log(f"New plot saved to: {save_path}")

    def _build_bias_plot(self, trainer: Trainer, epoch: int, dataset: str, **kwargs) -> None:
        # ensure correct kwargs are passed
        _required_kw = ["e_true"]
        for kw in _required_kw:
            if kw not in kwargs.keys():
                raise KeyError(f"Bias plotter requires setting the key word argument '{kw}'.")

        preds = trainer.last_eval[dataset]["preds"]
        targs = trainer.last_eval[dataset]["targets"]
        incls = trainer.last_eval[dataset]["includes"]

        e_true = kwargs["e_true"]
        if e_true in self._include_labels:
            x = incls[:, self._include_labels.index(e_true)].clone()
        elif e_true in self._target_labels:
            x = targs[:, self._target_labels.index(e_true)].clone()
        else:
            raise KeyError(f"Key '{e_true}' not found in target or included labels, you messed up!")

        xaxis_title = r"\text{%s}" % e_true
        if e_true in self._y_asinh_mask:
            x = torch.log10(x)
            xaxis_title = r"log_{10}\left[%s\right]" % xaxis_title

        n_cols = preds.shape[1]

        for i in range(n_cols):
            label = self._target_labels[i]
            yaxis_title = r"\text{(True - Reco)/True %s}" % label

            pred = preds[:, i]
            targ = targs[:, i]

            y = (targ - pred) / targ
            if e_true in self._y_asinh_mask:
                y = torch.log10(y)
                yaxis_title = r"log_{10}\left[%s\right]" % yaxis_title

            layout_kwargs = {
                "title": f"{label} Bias [Epoch {epoch + 1} - {dataset.title()}]",
                "yaxis_title": "$%s$" % yaxis_title,
                "xaxis_title": "$%s$" % xaxis_title
            }

            save_path = self._plot_dir(trainer) / f"{label}.bias.{epoch + 1}.html"

            plot = BiasPlot()
            plot.plot(
                x=x,
                y=y,
                save_path=save_path,
                layout_kwargs=layout_kwargs
            )
            trainer.console.log(f"New plot saved to: {save_path}")


class MulticlassMetricsCallback(Callback, PlotConfigurationMixin):

    _COMPATIBLE = ["multiclass"]
    _ALIASES: Dict[str, List[str]] = {
        "cm": [
            "confusion-matrix"
        ]
    }

    def __init__(self) -> None:
        self._target_label:     Optional[str]       = None
        self._include_labels:   Optional[List[str]] = None

        # plotting attrs
        self._plotters: List[Tuple[MetricsPlotMethod, Dict]] = []

        # dispatch dict
        self._dispatch: Dict[str, MetricsPlotMethod] = {
            "cm": self._build_confusion_matrix,
            "roc": self._build_roc_plot
        }

    def on_init(self, trainer: Trainer) -> None:
        self._target_label = IGData.attrs[0]["global"]["target_labels"][0]  # should be only one for multiclass

    def on_test_end(self, trainer: Trainer, epoch: int, metrics: ComputedMetrics) -> None:
        for plotter, kwargs in self._plotters:
            plotter(trainer, epoch, "test", **kwargs)

    def on_validation_end(self, trainer: Trainer, epoch: int, metrics: ComputedMetrics) -> None:
        for plotter, kwargs in self._plotters:
            plotter(trainer, epoch, "val", **kwargs)

    def _build_confusion_matrix(self, trainer: Trainer, epoch: int, dataset: str, **kwargs) -> None:
        preds = trainer.last_eval[dataset]["preds"]
        targs = trainer.last_eval[dataset]["targets"]

        layout_kwargs = {
            "title": f"{self._target_label} Confusion Matrix [Epoch {epoch + 1} - {dataset.title()}]",
            "yaxis_title": r"$\text{True %s}$" % self._target_label,
            "xaxis_title": r"$\text{Predicted %s}$" % self._target_label
        }

        save_path = self._plot_dir(trainer) / f"{self._target_label}.CM.{epoch + 1}.html"

        cm = ConfusionMatrixPlot()
        cm.plot(
            preds,
            targs,
            save_path=save_path,
            layout_kwargs=layout_kwargs
        )
        trainer.console.log(f"New plot saved to: {save_path}")

    def _build_roc_plot(self, trainer: Trainer, epoch: int, dataset: str, **kwargs) -> None:
        preds = trainer.last_eval[dataset]["preds"]
        targs = trainer.last_eval[dataset]["targets"]

        layout_kwargs = {
            "title": f"{self._target_label} ROC [Epoch {epoch + 1} - {dataset.title()}]"
        }

        save_path = self._plot_dir(trainer) / f"{self._target_label}.ROC.{epoch + 1}.html"

        roc = ROCPlot()
        roc.plot(
            preds,
            targs,
            save_path=save_path,
            layout_kwargs=layout_kwargs
        )
        trainer.console.log(f"New plot saved to: {save_path}")
