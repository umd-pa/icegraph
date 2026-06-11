# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

import plotly.graph_objects as go

from icegraph.common.histogram import Histogram

from ..plotter import HistogramPlotter1D

__all__ = ["Line2D"]


class Line2D(HistogramPlotter1D):

    def _plot_trace(self, fig: go.Figure, data: Histogram, label: str, **kwargs) -> None:
        fig.add_trace(
            go.Scatter(
                x=data.centers[0],
                y=data.histogram,
                mode="lines",
                name=label,
                showlegend=True,
                line=dict(color=kwargs.get("color")),
                legendgroup=label,
            )
        )

    def _apply_layout(self, fig: go.Figure, data: dict[str | int, Histogram], **kwargs) -> None:
        fig.update_yaxes(
            type="linear",
            autorange=True
        )
