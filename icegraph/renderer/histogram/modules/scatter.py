# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

import plotly.graph_objects as go

from icegraph.common.histogram import Histogram
from icegraph.renderer.style import PLOT_STYLE

from ..module import HistogramPlotterModule
from ..plotter import HistogramPlotter

__all__ = ["Scatter"]


class Scatter(HistogramPlotterModule):
    compatible = (HistogramPlotter,)

    def __init__(self, data: Histogram, name: str) -> None:
        self._overlay_data = data
        self._overlay_name = name

    def overlay(self, fig: go.Figure, data: dict[str | int, Histogram]) -> None:
        # grab centers from hist
        centers = self._overlay_data.centers[0]

        fig.add_trace(
            go.Scatter(
                x=centers,
                y=self._overlay_data.histogram,
                mode="markers",
                marker=dict(color=PLOT_STYLE.accent_1, size=8),
                name=self._overlay_name,
                showlegend=True
            )
        )