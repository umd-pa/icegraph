# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from abc import ABC, abstractmethod

from plotly import graph_objects as go

from icegraph.common.histogram import Histogram

from ..module import PlotterModule

__all__ = ["HistogramPlotterModule"]


class HistogramPlotterModule(PlotterModule, ABC):

    @abstractmethod
    def overlay(self, fig: go.Figure, data: dict[str | int, Histogram]) -> None:
        ...
