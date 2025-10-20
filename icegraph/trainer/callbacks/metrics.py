# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import Optional, TYPE_CHECKING, List, Dict, Tuple, Union
from pathlib import Path

import torch

from .callback import Callback
from icegraph.types import ComputedMetrics, MetricsPlotMethod
from icegraph.renderer import ParityPlot, BiasPlot, ConfusionMatrixPlot, ROCPlot, PRPlot

__all__ = [
    "RegressionMetricsCallback",
    "MulticlassMetricsCallback"
]

if TYPE_CHECKING:
    from .. import Trainer
else:
    class Trainer:
        pass


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
        self._y_asinh_mask =    trainer.registry.global_attrs["apply_log_scaling_y"]
        self._include_labels =  trainer.registry.global_attrs["include_labels"]
        self._target_labels =   trainer.registry.global_attrs["target_labels"]

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
        ],
        "pr": [
            "precision-recall"
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
            "roc": self._build_roc_plot,
            "pr": self._build_pr_plot
        }

    def on_init(self, trainer: Trainer) -> None:
        self._target_label = trainer.registry.global_attrs["target_labels"][0]  # should be only one for multiclass

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

    def _build_pr_plot(self, trainer: Trainer, epoch: int, dataset: str, **kwargs) -> None:
        preds = trainer.last_eval[dataset]["preds"]
        targs = trainer.last_eval[dataset]["targets"]

        layout_kwargs = {
            "title": f"{self._target_label} Precision Recall [Epoch {epoch + 1} - {dataset.title()}]"
        }

        save_path = self._plot_dir(trainer) / f"{self._target_label}.PR.{epoch + 1}.html"

        pr = PRPlot()
        pr.plot(
            preds,
            targs,
            save_path=save_path,
            layout_kwargs=layout_kwargs
        )
        trainer.console.log(f"New plot saved to: {save_path}")