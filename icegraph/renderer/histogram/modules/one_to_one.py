# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

import plotly.graph_objects as go
import numpy as np

from icegraph.common.histogram import Histogram
from icegraph.renderer.style import PLOT_STYLE

from ..module import HistogramPlotterModule
from ..plotter import HistogramPlotter

__all__ = ["OneToOne"]


class OneToOne(HistogramPlotterModule):
    compatible = (HistogramPlotter,)

    def overlay(self, fig: go.Figure, data: dict[str | int, Histogram]) -> None:
        # grab bounds from hist
        lo = np.min([h.mins.min() for h in data.values()])
        hi = np.max([h.maxs.max() for h in data.values()])

        # add 1:1 ideal line
        fig.add_trace(
            go.Scatter(
                x=[lo, hi],
                y=[lo, hi],
                mode="lines",
                line=dict(color=PLOT_STYLE.dark_gray, dash='dash', width=2),
                name="1:1",
                showlegend=True
            )
        )