# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing_extensions import override

import plotly.graph_objects as go
import numpy as np

from icegraph.common.histogram import Histogram
from icegraph.renderer.style import PLOT_STYLE

from ..plotter import HistogramPlotter2D

__all__ = ["Histogram2D"]


class Histogram2D(HistogramPlotter2D):

    @override
    def _plot_trace(self, fig: go.Figure, data: Histogram, label: str, **kwargs) -> None:
        # get centers
        centers = (np.arange(data.bins[0]), np.arange(data.bins[1])) if data.bounds is None else data.centers

        fig.add_trace(
            go.Heatmap(  # histogram
                x=centers[0],
                y=centers[1],
                z=data.histogram,
                zauto=False,
                zmin=0,
                coloraxis="coloraxis",
                showlegend=True,
                name=label,
                hoverongaps=False,
                legendgroup=label
            )
        )

    @override
    def _apply_layout(self, fig: go.Figure, data: dict[str | int, Histogram], **kwargs) -> None:
        if list(data.values())[0].bounds is not None:
            # get mins and maxs, find most min and most max among all histograms
            mins = np.minimum.reduce([h.mins for h in data.values()])
            maxs = np.maximum.reduce([h.maxs for h in data.values()])

        # make x and y-axes categorical if required
        else:
            mins = np.array([-0.5, -0.5])
            maxs = np.array(list(data.values())[0].bins) - 0.5

            fig.update_xaxes(dtick=1, tick0=0, showgrid=False, zeroline=False)
            fig.update_yaxes(dtick=1, tick0=0, showgrid=False, zeroline=False)

        # set plot bounds to data ranges
        fig.update_layout(
            xaxis_range=[mins[0], maxs[0]],
            yaxis_range=[mins[1], maxs[1]],
        )

        # allow plotly to plot on non-1:1 aspect ratio
        fig.update_xaxes(scaleanchor=None, constrain=None)
        fig.update_yaxes(scaleanchor=None, constrain=None)

        # get max value across all histograms
        max_value = max(float(h.peak_value) for h in data.values())

        # colorbar
        fig.update_layout(
            coloraxis=dict(
                cmin=0,
                cmax=max_value,
                colorscale=PLOT_STYLE.colorbar,
                colorbar=dict(title="Count", len=1, y=0.5)
            )
        )
