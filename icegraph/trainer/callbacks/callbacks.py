# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from datetime import datetime
from typing import Optional, TYPE_CHECKING, List, ClassVar, Callable, Dict, Any

import torch

from .base import Callback
from icegraph.console import Console
from icegraph.data.base import IGData
from icegraph.types import ComputedMetrics
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
    Trainer = None


class TensorBoardCallback(Callback):

    def __init__(self) -> None:
        self._tb: Optional[TensorBoard] = None

    def on_init(self, trainer) -> None:
        if trainer.strategy.task != "regression":
            raise NotImplementedError("Tensorboard callback is currently incompatible with non-regression strategies.")

        if self._tb is None:
            self._tb = TensorBoard(trainer.log_dir)
        self._tb.launch()

    def on_epoch_end(self, trainer, epoch, metrics) -> None:
        self._tb.writer.add_scalar("Train/MSE", metrics['loss'], epoch + 1)
        self._tb.writer.add_scalar("Train/RMSE", metrics['rmse'], epoch + 1)

    def on_validation_end(self, trainer, epoch, metrics) -> None:
        self._tb.writer.add_scalar("Validation/MSE", metrics['loss'], epoch + 1)
        self._tb.writer.add_scalar("Validation/RMSE", metrics['rmse'], epoch + 1)

    def on_test_end(self, trainer, epoch, metrics) -> None:
        self._tb.writer.add_scalar("Test/MSE", metrics['loss'], epoch + 1)
        self._tb.writer.add_scalar("Test/RMSE", metrics['rmse'], epoch + 1)

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
        display_metrics = [f"{metric.upper()}: {value:.4f}" for metric, value in metrics.items()]
        out = f" --> {' | '.join(display_metrics)}"

        Console.out(out)

    on_validation_end = on_test_end = on_epoch_end = display_metrics


class ExportCallback(Callback):
    def __init__(self) -> None:
        self._best_mse: float = float("inf")

    def _export(self, trainer: Trainer, epoch: int, metrics: ComputedMetrics) -> None:
        latest_path = trainer.outdir / "model_latest.pt"
        best_path = trainer.outdir / "model_best.pt"

        # Build CoreModel for export
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
            current_mse = metrics["loss"]
            if current_mse < self._best_mse:
                Console.out(
                    f"New best MSE {current_mse:.4f} < {self._best_mse:.4f}; "
                    f"saving best model to {best_path}...",
                    severity=1
                )
                self._best_mse = current_mse
                try:
                    torch.save(export_model, best_path)
                except Exception as e:
                    Console.out(f"Failed to save model: {e}", severity=3)

    # run both on validation and test, not on train
    on_validation_end = on_test_end = _export


class RegressionMetricsCallback(Callback):

    # class vars
    _plotters:  ClassVar[List[Callable[..., None]]] = []

    # define a cache for storing plotting configurations
    _cache:     ClassVar[Dict[str, Any]]            = {}

    def __init__(self) -> None:
        self._y_asinh_mask:     Optional[List[str]] = None
        self._target_labels:    Optional[List[str]] = None
        self._include_labels:   Optional[List[str]] = None

    def on_init(self, trainer: Trainer) -> None:
        self._y_asinh_mask =    IGData.attrs[0]["global"]["apply_log_scaling_y"]
        self._include_labels =  IGData.attrs[0]["global"]["include_labels"]
        self._target_labels =   IGData.attrs[0]["global"]["target_labels"]

    def on_test_end(self, trainer: Trainer, epoch: int, metrics: ComputedMetrics) -> None:
        for plotter in type(self)._plotters:
            plotter(self, trainer, epoch, "test")

    def on_validation_end(self, trainer: Trainer, epoch: int, metrics: ComputedMetrics) -> None:
        for plotter in type(self)._plotters:
            plotter(self, trainer, epoch, "val")

    @classmethod
    def enqueue_parity(cls) -> None:
        cls._plotters.append(cls._build_parity_plot)

    @classmethod
    def enqueue_bias(cls, e_true: str) -> None:
        cls._plotters.append(cls._build_bias_plot)

        # grab target and included labels from metadata
        _include_labels = IGData.attrs[0]["global"]["include_labels"]
        _target_labels = IGData.attrs[0]["global"]["target_labels"]

        # ensure the passed label exists in the processed dataset
        if e_true not in (_include_labels + _target_labels):
            raise KeyError(
                f"Label '{e_true}' not found in dataset, select from available labels: "
                f"[{', '.join(_include_labels + _target_labels)}]"
            )

        # if it exists, cache it for the plotter
        cls._cache["e_true"] = e_true

    def _build_parity_plot(self, trainer: Trainer, epoch: int, dataset: str) -> None:
        preds = getattr(trainer, f"{dataset}_predictions")
        targs = getattr(trainer, f"{dataset}_targets")

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

    def _build_bias_plot(self, trainer: Trainer, epoch: int, dataset: str) -> None:
        cls = type(self)

        preds: torch.Tensor = getattr(trainer, f"{dataset}_predictions")
        targs: torch.Tensor = getattr(trainer, f"{dataset}_targets")
        incls: torch.Tensor = getattr(trainer, f"{dataset}_includes")

        e_true = cls._cache["e_true"]
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


class MulticlassMetricsCallback(Callback):

    # class vars
    _plotters:  ClassVar[List[Callable[..., None]]] = []

    def __init__(self) -> None:
        self._target_label:     Optional[str]       = None
        self._include_labels:   Optional[List[str]] = None

    def on_init(self, trainer: Trainer) -> None:
        self._include_labels =  IGData.attrs[0]["global"]["include_labels"]
        self._target_label =   IGData.attrs[0]["global"]["target_labels"][0]  # should be only one for multiclass

    def on_test_end(self, trainer: Trainer, epoch: int, metrics: ComputedMetrics) -> None:
        for plotter in type(self)._plotters:
            plotter(self, trainer, epoch, "test")

    def on_validation_end(self, trainer: Trainer, epoch: int, metrics: ComputedMetrics) -> None:
        for plotter in type(self)._plotters:
            plotter(self, trainer, epoch, "val")

    @classmethod
    def enqueue_confusion_matrix(cls) -> None:
        cls._plotters.append(cls._build_confusion_matrix)

    @classmethod
    def enqueue_roc(cls) -> None:
        cls._plotters.append(cls._build_roc_plot)

    def _build_confusion_matrix(self, trainer: Trainer, epoch: int, dataset: str) -> None:
        preds: torch.Tensor = getattr(trainer, f"{dataset}_predictions")
        targs: torch.Tensor = getattr(trainer, f"{dataset}_targets")

        layout_kwargs = {
            "title": f"{self._target_label} Confusion Matrix [Epoch {epoch + 1} - {dataset.title()}]",
            "yaxis_title": r"$\text{True %s}$" % self._target_label,
            "xaxis_title": r"$\text{Predicted %s}$" % self._target_label
        }

        cm = ConfusionMatrixPlot()
        cm.plot(
            preds,
            targs,
            save_path=trainer.outdir / f"{self._target_label}.cm.{epoch + 1}.html",
            layout_kwargs=layout_kwargs
        )

    def _build_roc_plot(self, trainer: Trainer, epoch: int, dataset: str) -> None:
        preds: torch.Tensor = getattr(trainer, f"{dataset}_predictions")
        targs: torch.Tensor = getattr(trainer, f"{dataset}_targets")

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
