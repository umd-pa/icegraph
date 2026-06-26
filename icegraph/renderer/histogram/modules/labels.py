# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing_extensions import override

import plotly.graph_objects as go
import numpy as np

from icegraph.common.histogram import Histogram

from ..module import HistogramPlotterModule
from ..plotter import HistogramPlotter2D

__all__ = ["Labels"]


class Labels(HistogramPlotterModule):
    compatible = (HistogramPlotter2D,)

    @override
    def overlay(self, fig: go.Figure, data: dict[str | int, Histogram]) -> None:
        for i, (key, item) in enumerate(data.items()):
            self._overlay_trace(fig, item, str(key))

    @staticmethod
    def _overlay_trace(fig: go.Figure, data: Histogram, label: str) -> None:
        centers = (np.arange(data.bins[0]), np.arange(data.bins[1])) if data.bounds is None else data.centers

        text_x = []
        text_y = []
        text = []

        if data.bounds is None:
            dx = dy = 0.45
        else:
            dx, dy = 0.45 * data.widths

        for (iy, ix), cell_value in np.ndenumerate(data.histogram):
            text_x.append(centers[0][ix] - dx)
            text_y.append(centers[1][iy] - dy)
            text.append(f"{cell_value:.3g}")

        fig.add_trace(
            go.Scatter(
                x=text_x,
                y=text_y,
                text=text,
                mode="text",
                textposition="middle right",
                textfont=dict(color="black", size=20),
                hoverinfo="skip",
                showlegend=False,
                legendgroup=label
            )
        )