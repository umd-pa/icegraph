# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union

import plotly.graph_objects as go

from icegraph.common.histogram import Histogram

__all__ = ["Plotter"]


class Plotter(ABC):
    """Base class for IceGraph plots."""
    # plot sizing
    _inner_px = 600
    _pad_px = 80

    # for any colorbar (low saturation steel blue)
    _colorbar = [[0, "rgba(0, 0, 0, 0)"], [0.0001, "#B5CAE6"], [1, "#08519C"]]

    # plot base colors
    _background_color = "#FFFFFF"
    _border_color = "#111111"
    _legend_border_color = "rgba(0, 0, 0, 0.25)"
    _legend_background_color = "rgba(255, 255, 255, 0.7)"
    _grid_color = "#CCCCCC"

    # for simple overlays
    _dark_gray = '#222222'

    # base color (steel blue)
    _color_1 = "#B5CAE6"

    # accent 1 (magenta) variations
    _accent_1 = 'rgba(255,45,134,1)'
    _accent_1_opaque = 'rgba(255,45,134,0.85)'
    _accent_1_opaque_fill = 'rgba(255,45,134,0.18)'

    def __init__(self, hist: Histogram, *, epoch: int | None = None) -> None:
        # init global fig object
        self._fig = go.Figure()

        # slot for histogram and epoch if passed
        self._hist:     Histogram   = hist
        self._epoch:    int | None  = epoch

    def _apply_layout(self, **kwargs) -> None:
        # set font
        self._fig.update_layout(font=dict(size=16))

        # set plot background colors
        self._fig.update_layout(plot_bgcolor=self._background_color, paper_bgcolor=self._background_color)

        # design the legend
        self._fig.update_layout(
            legend=dict(
                x=0.02, y=0.98, xanchor="left", yanchor="top",
                bgcolor=self._legend_background_color, bordercolor=self._legend_border_color,
                borderwidth=1, orientation="v"
            )
        )

        # set plot size and margins
        self._fig.update_layout(
            width=self._inner_px + self._pad_px + self._pad_px,
            height=self._inner_px + self._pad_px + self._pad_px,
            margin=dict(l=self._pad_px, r=self._pad_px, t=self._pad_px, b=self._pad_px)
        )

        # format the axes (same for both x and y)
        axis_kwargs = dict(
            showgrid=True, gridcolor=self._grid_color, gridwidth=1, zeroline=False,
            showline=True, linecolor=self._border_color, linewidth=1.25, mirror=True
        )

        self._fig.update_xaxes(**axis_kwargs)
        self._fig.update_yaxes(**axis_kwargs)

        # lock paperspace aspect ratio
        self._fig.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1))

        # force scientific notation for axis ticks
        self._fig.update_layout(
            yaxis=dict(
                exponentformat='e', showexponent='all'
            ),
            xaxis = dict(
                exponentformat='e', showexponent='all'
            )
        )

        # apply subclass-specific layout options
        self._apply_subclass_layout(self._fig, self._hist)

        # apply kwargs last so they can override other selections
        self._fig.update_layout(**kwargs)

    def plot(self, save_path: Union[str, Path], **kwargs) -> None:
        save_path = Path(save_path)

        # execute subclass plot logic
        assert self._hist is not None, "Histogram not loaded."
        self._plot(self._fig, self._hist)

        # apply default plot layout
        self._apply_layout(**kwargs)

        # write to html
        self._fig.write_html(
            str(save_path),
            config={
                "toImageButtonOptions": {
                    "filename": save_path.with_suffix("").name
                }
            },
            full_html=True,
            include_plotlyjs="cdn",
            include_mathjax="cdn"
        )

    def _apply_subclass_layout(self, fig: go.Figure, hist: Histogram) -> None:
        # default: do nothing
        pass

    @abstractmethod
    def _plot(self, fig: go.Figure, hist: Histogram) -> None: ...
