# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from abc import ABC, abstractmethod

from plotly import graph_objects as go

from icegraph.common.histogram import Histogram
from icegraph.renderer.plotter import Plotter
from icegraph.renderer.style import PLOT_STYLE

from .module import HistogramPlotterModule

__all__ = ["HistogramPlotter", "HistogramPlotter1D", "HistogramPlotter2D"]


class HistogramPlotter(Plotter[dict[str | int, Histogram], HistogramPlotterModule], ABC):

    def ensure_ndim(self, data: dict[str | int, Histogram], ndim: int) -> None:
        # ensure all histograms are correct dim
        for key, item in data.items():
            if item.histogram.ndim != ndim:
                raise ValueError(
                    f"{type(self).__name__}.plot expected a dict of {ndim}D histograms, "
                    f"got shape {item.histogram.shape} for item '{key}'."
                )

    def ensure_bin_count(self, data: dict[str | int, Histogram]) -> None:
        # ensure each histogram has the same bin count
        expected_bins: tuple[int, ...] | None = None

        for key, item in data.items():
            bins = item.bins

            if expected_bins is None:
                expected_bins = bins
                continue

            if bins != expected_bins:
                raise ValueError(
                    f"{type(self).__name__}.plot expected all histograms to have the same "
                    f"bin count, got bins={bins} for item '{key}' instead of "
                    f"bins={expected_bins}."
                )

    def _plot(self, fig: go.Figure, data: dict[str | int, Histogram], **kwargs) -> None:
        for i, (key, item) in enumerate(data.items()):
            color = PLOT_STYLE.theme_sequence[i % len(PLOT_STYLE.theme_sequence)]

            self._plot_trace(fig, item, str(key), color=color)

    @abstractmethod
    def _plot_trace(self, fig: go.Figure, data: Histogram, label: str, **kwargs) -> None:
        ...


class HistogramPlotter2D(HistogramPlotter, ABC):

    def _validate_data(self, data) -> None:
        # ensure all histograms are 2D and have identical bin count
        self.ensure_ndim(data, 2)
        self.ensure_bin_count(data)


class HistogramPlotter1D(HistogramPlotter, ABC):

    def _validate_data(self, data) -> None:
        # ensure all histograms are 1D and have identical bin count
        self.ensure_ndim(data, 1)
        self.ensure_bin_count(data)
