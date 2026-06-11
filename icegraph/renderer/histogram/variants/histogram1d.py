# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

import plotly.graph_objects as go

from icegraph.common.histogram import Histogram

from ..plotter import HistogramPlotter1D

__all__ = ["Histogram1D"]


class Histogram1D(HistogramPlotter1D):

    def _plot_trace(self, fig: go.Figure, data: Histogram, label: str, **kwargs) -> None:
        fig.add_trace(
            go.Bar(
                x=data.centers[0],
                y=data.histogram,
                width=0.95 * data.widths[0],
                name=label,
                showlegend=True,
                marker=dict(color=kwargs.get("color")),
                opacity=0.7,
                legendgroup=label
            )
        )

    def _apply_layout(self, fig: go.Figure, data: dict[str | int, Histogram], **kwargs) -> None:
        # get min and max values along x
        lo = min([d.mins[0] for d in data.values()])
        hi = max([d.maxs[0] for d in data.values()])

        fig.update_yaxes(
            type="linear",
            autorange=True,
            rangemode="tozero"
        )

        fig.update_xaxes(
            range=[lo, hi],
            autorange=False,
        )

        # make x-axis categorical if required
        if kwargs.get("categorical", False):
            nx = list(data.values())[0].bins[0]

            fig.update_xaxes(
                type="category",
                categoryorder="array",
                categoryarray=list(range(nx)),
                scaleanchor=None,
                constrain=None,
            )

        fig.update_layout(barmode="overlay")
