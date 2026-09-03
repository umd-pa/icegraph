# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing_extensions import override

import plotly.graph_objects as go
import numpy as np

from icegraph.common.histogram import Histogram
from icegraph.renderer.style import PLOT_STYLE

from ..module import HistogramPlotterModule
from ..plotter import HistogramPlotter

__all__ = ["VLine"]


class VLine(HistogramPlotterModule):
    compatible = (HistogramPlotter,)

    def __init__(self, **kwargs) -> None:
        self._kwargs = kwargs

    @override
    def overlay(self, fig: go.Figure, data: dict[str | int, Histogram]) -> None:
        fig.add_vline(**self._kwargs)
