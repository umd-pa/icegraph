# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import Optional, Sequence, Union

import numpy as np
import plotly.graph_objects as go

from .base import IGBasicPlot, IGDistributionPlot
from icegraph.data.pulses import Pulses
from icegraph.exceptions import IceCubeImportError

import warnings

# Silence Boost.Python converter warnings
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message=r".*to-Python converter for.*already registered.*"
)

try:
    from icecube.icetray import OMKey as _OMKey
except ImportError:
    _OMKey = IceCubeImportError.IceCubeMissingBase

OMKey = _OMKey

__all__ = ["CDFPlot", "PDFPlot", "ChargeDistributionPlot", "ParityPlot"]


class CDFPlot(IGDistributionPlot):

    plot_type = "CDF"

    def _populate_plot(self, target_dom: Optional[OMKey] = None) -> Pulses.DOMMetadata:
        times, cdf, metadata = self._pulses.get_cdf(target_dom)

        self._fig.add_trace(go.Scatter(x=times, y=cdf, mode="lines", name="CDF", line=dict(width=4)))
        self._fig.update_layout(
            xaxis_title="Time [ns]",
            yaxis_title="Normalized Cumulative Charge"
        )

        return metadata


class PDFPlot(IGDistributionPlot):

    plot_type = "PDF"

    def _populate_plot(self, target_dom: Optional[OMKey] = None) -> Pulses.DOMMetadata:
        times, charges, metadata = self._pulses.get_pulses(target_dom)

        # Compute weighted histogram
        min_t = np.floor(min(times))
        max_t = min_t + 500  # avoid any noise and cut to 0.5 microsecond
        bin_edges = np.linspace(min_t, max_t, 51)
        hist_vals, bins = np.histogram(times, bins=bin_edges, weights=charges)

        bin_centers = 0.5 * (bins[1:] + bins[:-1])

        self._fig.add_trace(go.Bar(
            x=bin_centers,
            y=hist_vals,
            width=np.diff(bins),
            name='PDF'
        ))

        self._fig.update_layout(
            xaxis_title="Time [ns]",
            yaxis_title="Charge [PE]",
            bargap=0.1
        )

        return metadata


class ChargeDistributionPlot(IGDistributionPlot):

    plot_type = "charge_dist"

    def _populate_plot(self, target_dom: Optional[OMKey] = None) -> Pulses.DOMMetadata:
        _, charges, metadata = self._pulses.get_pulses(target_dom)

        self._fig.add_trace(go.Histogram(x=charges, nbinsx=100, name="CD"))
        self._fig.update_layout(
            xaxis_title="Charge [PE]",
            yaxis_title="Count",
            bargap=0.1
        )

        return metadata


class ParityPlot(IGBasicPlot):

    def _populate_plot(self, x: Sequence[Union[int, float]], y: Sequence[Union[int, float]]) -> None:
        _range = [min(x), max(x)]
        self._fig.update_layout(
            xaxis_range=_range,
            yaxis_range=_range,
        )
        self._fig.add_trace(go.Histogram2d(
            x=x, y=y,
            nbinsx=100, nbinsy=100,
            xbins=dict(start=_range[0], end=_range[1], size=(_range[1] - _range[0]) / 100),
            ybins=dict(start=_range[0], end=_range[1], size=(_range[1] - _range[0]) / 100),
            colorscale=[[0.0, "black"], [0.0001, "red"], [0.5, "yellow"], [1, "white"]],
            zmin=0,
            colorbar=dict(title="Count")
        ))
        self._fig.add_trace(go.Scatter(x=_range, y=_range, mode="lines", line=dict(color='green', dash='dash', width=5)))