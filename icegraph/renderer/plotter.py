# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Generic, TypeVar, TYPE_CHECKING
from abc import ABC, abstractmethod
from pathlib import Path

import plotly.graph_objects as go

from .style import PLOT_STYLE

if TYPE_CHECKING:
    from .module import PlotterModule

__all__ = ["Plotter"]


T = TypeVar("T")
M = TypeVar("M", bound="PlotterModule")


class Plotter(ABC, Generic[T, M]):
    """Base class for IceGraph plots."""

    def __init__(self) -> None:
        # init global fig object
        self._fig = go.Figure()

        # cache modules
        self._modules: list[M] = []

        # default legend location
        self._legend_location = {
            "x": 0.02, "y": 0.98, "xanchor": "left", "yanchor": "top"
        }

    def set_title(self, title: str) -> None:
        self._fig.update_layout(
            title_text=title,
            title_font=dict(size=20)
        )

    def set_xlabel(self, label: str) -> None:
        self._fig.update_xaxes(
            title_text=label,
            title_font=dict(size=24)
        )

    def set_ylabel(self, label: str) -> None:
        self._fig.update_yaxes(
            title_text=label,
            title_font=dict(size=24)
        )

    def set_legend_location(self, x: float, y: float, xanchor: str, yanchor: str) -> None:
        self._legend_location["x"] = x
        self._legend_location["y"] = y
        self._legend_location["xanchor"] = xanchor
        self._legend_location["yanchor"] = yanchor


    def add_module(self, module: M | list[M]) -> None:
        # recursion
        if isinstance(module, list):
            for m in module:
                self.add_module(m)

        # make sure a list didnt leak through
        assert not isinstance(module, list)

        # validate and append
        module.validate_for(self)
        self._modules.append(module)

    def _apply_default_layout(self, fig: go.Figure) -> None:
        # set font
        fig.update_layout(font=dict(size=16))

        # set plot background colors
        fig.update_layout(plot_bgcolor=PLOT_STYLE.background_color, paper_bgcolor=PLOT_STYLE.background_color)

        # design the legend
        fig.update_layout(
            legend=dict(
                **self._legend_location,
                bgcolor=PLOT_STYLE.legend_background_color, bordercolor=PLOT_STYLE.legend_border_color,
                borderwidth=1, orientation="v"
            )
        )

        # set plot size and margins
        padding = PLOT_STYLE.pad_px

        fig.update_layout(
            width=PLOT_STYLE.inner_px + 2 * padding,
            height=PLOT_STYLE.inner_px + 2 * padding,
            margin=dict(l=padding, r=padding, t=padding, b=padding),
        )

        # format the axes (same for both x and y)
        axis_kwargs = dict(
            showgrid=True, gridcolor=PLOT_STYLE.grid_color, gridwidth=1, zeroline=False,
            showline=True, linecolor=PLOT_STYLE.border_color, linewidth=1.25, mirror=True
        )

        fig.update_xaxes(**axis_kwargs)  # pyright: ignore[reportArgumentType]
        fig.update_yaxes(**axis_kwargs)  # pyright: ignore[reportArgumentType]

        # force scientific notation for axis ticks
        fig.update_xaxes(
            exponentformat="e",
            showexponent="all",
        )
        fig.update_yaxes(
            exponentformat="e",
            showexponent="all",
        )

    def plot(self, data: T, path: str | Path, **kwargs) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)  # ensure path exists before save

        # apply default plot layout
        self._apply_default_layout(self._fig)

        # apply subclass specific layout
        self._apply_layout(self._fig, data, **kwargs)

        # execute subclass plot logic
        self._plot(self._fig, data, **kwargs)

        # plot all overlays
        for module in self._modules:
            module.overlay(self._fig, data)

        # write to html
        self._fig.write_html(
            str(path),
            config={
                "toImageButtonOptions": {
                    "filename": path.with_suffix("").name
                }
            },
            full_html=True,
            include_plotlyjs="cdn",
            include_mathjax="cdn"
        )

    @abstractmethod
    def _validate_data(self, data) -> None:
        ...

    @abstractmethod
    def _apply_layout(self, fig: go.Figure, data: T, **kwargs) -> None:
        ...

    @abstractmethod
    def _plot(self, fig: go.Figure, data: T, **kwargs) -> None:
        ...
