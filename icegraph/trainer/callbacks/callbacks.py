# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from datetime import datetime
from typing import Optional, TYPE_CHECKING, List, Dict, Tuple, Union

import torch

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


class TensorBoardCallback(Callback):

    def __init__(self) -> None:
        self._tb: Optional[TensorBoard] = None

    def on_init(self, trainer) -> None:
        if self._tb is None:
            self._tb = TensorBoard(trainer.log_dir)
        self._tb.launch()

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


class ConsoleCallback(Callback):

    def on_train_begin(self, trainer) -> None:
        Console.out(f"Trainer output directory: {trainer.outdir}")

        # warn if falling back to CPU
        if trainer.device.type == "cpu":
            Console.out("No accelerators found, falling back to CPU training.", severity=2)

    def on_epoch_begin(self, trainer, epoch) -> None:
        Console.out(f"[Train] Epoch {epoch + 1}/{trainer.trainer_config.max_epochs}")

    def on_validation_begin(self, trainer, epoch) -> None:
        Console.out(f"[Validation] Epoch {epoch + 1}/{trainer.trainer_config.max_epochs}")

    def on_test_begin(self, trainer, epoch) -> None:
        Console.out(f"[Test] Epoch {epoch + 1}/{trainer.trainer_config.max_epochs}")

    def display_metrics(self, trainer, epoch, metrics) -> None:
        # task agnostic display of metrics
        display_metrics = [f"{metric.upper().split(':')[-1]}: {value:.4f}" for metric, value in metrics.items()]
        out = f" --> {' | '.join(display_metrics)}"

        Console.out(out)

    on_validation_end = on_test_end = on_epoch_end = display_metrics


class ExportCallback(Callback):
    def __init__(self) -> None:
        self._best_loss: float = float("inf")

    def _export(self, trainer: Trainer, epoch: int, metrics: ComputedMetrics) -> None:
        latest_path = trainer.outdir / "model_latest.pt"
        best_path = trainer.outdir / "model_best.pt"

        # Build CoreModel for export
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

        label = f"[Epoch {epoch + 1}]" if epoch is not None else ""
        Console.out(f"{label} Saving latest model to {latest_path}...")

        try:
            torch.save(export_model, latest_path)
        except Exception as e:
            Console.out(f"Failed to save model: {e}", severity=3)

        # Save best model if improved
        if metrics is not None:
            loss_key = next((k for k in metrics if k.startswith("loss")), None)
            loss = metrics[loss_key]
            if loss < self._best_loss:
                Console.out(
                    f"New best {loss_key.split(':')[1].upper()} {loss:.4f} < {self._best_loss:.4f}; "
                    f"saving best model to {best_path}...",
                    severity=1
                )
                self._best_loss = loss
                try:
                    torch.save(export_model, best_path)
                except Exception as e:
                    Console.out(f"Failed to save model: {e}", severity=3)

    # run both on validation and test, not on train
    on_validation_end = on_test_end = _export


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

            plot = ParityPlot()
            plot.plot(
                x=targ,
                y=pred,
                save_path=trainer.outdir / f"{label}.parity.{epoch + 1}.html",
                layout_kwargs=layout_kwargs
            )

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

            plot = BiasPlot()
            plot.plot(
                x=x,
                y=y,
                save_path=trainer.outdir / f"{label}.bias.{epoch + 1}.html",
                layout_kwargs=layout_kwargs
            )


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

        cm = ConfusionMatrixPlot()
        cm.plot(
            preds,
            targs,
            save_path=trainer.outdir / f"{self._target_label}.CM.{epoch + 1}.html",
            layout_kwargs=layout_kwargs
        )

    def _build_roc_plot(self, trainer: Trainer, epoch: int, dataset: str, **kwargs) -> None:
        preds = trainer.last_eval[dataset]["preds"]
        targs = trainer.last_eval[dataset]["targets"]

        layout_kwargs = {
            "title": f"{self._target_label} ROC [Epoch {epoch + 1} - {dataset.title()}]"
        }

        cm = ROCPlot()
        cm.plot(
            preds,
            targs,
            save_path=trainer.outdir / f"{self._target_label}.ROC.{epoch + 1}.html",
            layout_kwargs=layout_kwargs
        )
