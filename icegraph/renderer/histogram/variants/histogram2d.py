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


# legend acts as a radio group over heatmaps
_EXCLUSIVE_LEGEND_JS = """
var gd = document.getElementById('{plot_id}');

gd.on('plotly_legendclick', function(ev) {
    if (ev.data[ev.curveNumber].type !== 'heatmap') return true;

    var indices = [], visible = [];
    ev.data.forEach(function(trace, i) {
        if (trace.type !== 'heatmap') return;
        indices.push(i);
        visible.push(i === ev.curveNumber ? true : 'legendonly');
    });

    Plotly.update(gd, {visible: visible}, {'coloraxis.cmax': ev.data[ev.curveNumber].meta}, indices);
    return false;
});

gd.on('plotly_legenddoubleclick', function(ev) {
    return ev.data[ev.curveNumber].type !== 'heatmap';
});
"""


class Histogram2D(HistogramPlotter2D):
    """
    2D heatmap plotter. With ``exclusive``, only one series is shown at a time (the first
    by default).
    """

    def __init__(self, exclusive: bool = False) -> None:
        super().__init__()

        self._exclusive = exclusive

        if exclusive:
            self._post_scripts.append(_EXCLUSIVE_LEGEND_JS)

    @override
    def _plot_trace(self, fig: go.Figure, data: Histogram, label: str, **kwargs) -> None:
        # get centers
        centers = (np.arange(data.bins[0]), np.arange(data.bins[1])) if data.bounds is None else data.centers

        # in exclusive mode, only the first heatmap starts visible
        first = not any(trace.type == "heatmap" for trace in fig.data)
        visible = True if first or not self._exclusive else "legendonly"

        fig.add_trace(
            go.Heatmap(  # histogram
                x=centers[0],
                y=centers[1],
                z=data.histogram.T,  # histogram is [x, y], plotly wants z rows as y
                zauto=False,
                zmin=0,
                coloraxis="coloraxis",
                showlegend=True,
                name=label,
                hoverongaps=False,
                legendgroup=label,
                visible=visible,
                meta=float(data.peak_value) if self._exclusive else None
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

        # get max value across all histograms, or just the initially visible one if exclusive
        peaks = [float(h.peak_value) for h in data.values()]
        max_value = peaks[0] if self._exclusive else max(peaks)

        # colorbar
        fig.update_layout(
            coloraxis=dict(
                cmin=0,
                cmax=max_value,
                colorscale=PLOT_STYLE.colorbar,
                colorbar=dict(title="Count", len=1, y=0.5)
            )
        )
