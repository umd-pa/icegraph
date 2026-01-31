# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from plotly import graph_objects as go

from .median_quantile import MedianQuantileBand
from icegraph.common.histogram import Histogram

__all__ = ["ParityPlot"]


class ParityPlot(MedianQuantileBand):

    def _plot_overlay(self, fig: go.Figure, hist: Histogram) -> None:
        # grab bounds from hist
        ranges = hist.bounds

        # add 1:1 ideal line
        fig.add_trace(go.Scatter(
            x=ranges[0], y=ranges[1], mode="lines",
            line=dict(color=self._dark_gray, dash='dash', width=2), name="Ideal",
            showlegend=True
        ))
