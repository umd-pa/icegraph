# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import Optional, Sequence, Union, Tuple

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

__all__ = ["CDFPlot", "PDFPlot", "ChargeDistributionPlot", "ParityPlot", "BiasPlot"]


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

    def _populate_plot(self, x: np.ndarray, y: np.ndarray, nbins: int) -> None:
        _range = [min(x), max(x)]

        self._fig.update_layout(
            xaxis_range=_range,
            yaxis_range=_range,
        )

        edges = np.linspace(_range[0], _range[1], nbins + 1)
        H, _, _ = np.histogram2d(x, y, bins=[edges, edges])

        Z = H.astype(int)

        centers = (edges[:-1] + edges[1:]) / 2
        zmax = np.nanmax(Z) if np.isfinite(np.nanmax(Z)) else 1.0

        # plot
        self._fig.update_layout(
            coloraxis=dict(
                cmin=0, cmax=float(zmax),
                colorscale=self._colorbar,
                colorbar=dict(title="Count", len=1, y=0.5)
            )
        )
        self._fig.add_trace(go.Heatmap(
            x=centers, y=centers, z=Z.T,
            zauto=False, zmin=0,
            coloraxis="coloraxis",
            showlegend=True, name="Data",
            hoverongaps=False
        ))

        # calculate binned median and containment
        edges, m_e, lq_e, uq_e = self.get_median_and_containment(x, y, int(nbins / 2))

        # plot containment
        self._fig.add_trace(go.Scatter(
            x=edges, y=lq_e, mode='lines',
            line=dict(color=self._accent_1_opaque, width=1, dash='dot'),
            showlegend=False, line_shape='hv',
            legendgroup="median"
        ))
        self._fig.add_trace(go.Scatter(
            x=edges, y=uq_e, mode='lines',
            line=dict(color=self._accent_1_opaque, width=1, dash='dot'),
            fill='tonexty', fillcolor=self._accent_1_opaque_fill,
            showlegend=False, line_shape='hv',
            legendgroup="median"
        ))

        # add median line
        self._fig.add_trace(go.Scatter(
            x=edges, y=m_e, mode='lines',
            line=dict(color="white", width=3),
            showlegend=False, legendgroup="median",
            line_shape='hv'
        ))
        self._fig.add_trace(go.Scatter(
            x=edges, y=m_e, mode='lines',
            line=dict(color=self._accent_1, width=2), name='Median w/ 68% Containment',
            showlegend=True, legendgroup="median",
            line_shape='hv'
        ))

        # plot 1:1 line
        self._fig.add_trace(go.Scatter(
            x=_range, y=_range, mode="lines",
            line=dict(color=self._dark_gray, dash='dash', width=2), name="1:1",
            showlegend=True
        ))

        # lock paperspace aspect ratio
        self._fig.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1))

    @staticmethod
    def get_median_and_containment(
            x: Sequence[Union[int, float]],
            y: Sequence[Union[int, float]],
            nbins: int = 50,
            containment: float = 0.68
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate binned median and containment as a stepwise series.

        Returns:
            edges: (nbins+1) bin edges over x
            med_e: (nbins+1) median values edge-aligned
            lo_e:  (nbins+1) lower quantile edge-aligned
            hi_e:  (nbins+1) upper quantile edge-aligned
        """
        # allow torch tensors
        if hasattr(x, "detach"): x = x.detach().cpu().numpy()
        if hasattr(y, "detach"): y = y.detach().cpu().numpy()

        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()

        # only care about finite values, mask out non finite
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]

        # if there are no finite values, just return all nans (this likely wont happen)
        if x.size == 0:
            edges = np.linspace(0.0, 1.0, nbins + 1)
            nanline = np.full(nbins + 1, np.nan)
            return edges, nanline, nanline, nanline

        # ensure containment between 0 and 1 and define quartiles
        containment = float(min(max(containment, 0.0), 1.0))
        q_lo_p, q_med_p, q_hi_p = (0.5 - containment / 2, 0.5, 0.5 + containment / 2)

        # bins on x
        edges = np.linspace(x.min(), x.max(), nbins + 1)

        med, lo, hi = [], [], []
        for i in range(nbins):
            lo_edge, hi_edge = edges[i], edges[i + 1]

            # build mask to select values within the bin region
            if i < nbins - 1:
                mask = (x >= lo_edge) & (x < hi_edge)
            else:
                # include right edge on the last bin so max(x) is captured
                mask = (x >= lo_edge) & (x <= hi_edge)

            if np.any(mask):
                # ignore nans in y within the bin
                vals = y[mask]
                q_lo, q_med, q_hi = np.nanquantile(vals, [q_lo_p, q_med_p, q_hi_p])
                lo.append(q_lo)
                med.append(q_med)
                hi.append(q_hi)
            else:
                # append nans if no values in bin region
                lo.append(np.nan)
                med.append(np.nan)
                hi.append(np.nan)

        med = np.asarray(med, dtype=float)
        lo = np.asarray(lo, dtype=float)
        hi = np.asarray(hi, dtype=float)

        # repeat last value so length = nbins + 1
        med_e = np.r_[med, med[-1]]
        lo_e = np.r_[lo, lo[-1]]
        hi_e = np.r_[hi, hi[-1]]

        return edges, med_e, lo_e, hi_e


class BiasPlot(IGBasicPlot):

    def _populate_plot(self, x: np.ndarray, y: np.ndarray, nbins: int) -> None:
        # plot
        self._fig.add_trace(go.Scatter(
            x=x, y=y, mode="markers"
        ))