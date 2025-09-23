# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import Optional, Sequence, Union, Tuple, Dict, Any
import math

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

__all__ = ["CDFPlot", "PDFPlot", "ChargeDistributionPlot", "ParityPlot", "BiasPlot", "ConfusionMatrixPlot", "ROCPlot"]


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


class AnalysisMixin:

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

    @staticmethod
    def stairs(edges: np.ndarray, vals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x = np.repeat(edges, 2)[1:-1]
        y = np.repeat(vals, 2)
        return x, y

    def add_step_polygon(
            self,
            fig: go.Figure, *,
            edges: np.ndarray,
            lower: np.ndarray,
            upper: np.ndarray,
            fill_color: str,
            outline_color: str,
            legendgroup: Optional[str] = None
    ) -> None:
        edges = np.asarray(edges, float)
        lower = np.asarray(lower, float)
        upper = np.asarray(upper, float)
        valid = np.isfinite(lower) & np.isfinite(upper)
        i, n = 0, lower.size
        while i < n:
            if not valid[i]:
                i += 1;
                continue
            j = i
            while j < n and valid[j]:
                j += 1
            x_lo, y_lo = self.stairs(edges[i:j + 1], lower[i:j])
            x_up, y_up = self.stairs(edges[i:j + 1], upper[i:j])
            fig.add_trace(go.Scatter(
                x=np.concatenate([x_lo, x_up[::-1]]),
                y=np.concatenate([y_lo, y_up[::-1]]),
                mode='lines', line=dict(width=0),
                fill='toself', fillcolor=fill_color,
                hoverinfo='skip', showlegend=False,
                legendgroup=legendgroup
            ))
            # outlines
            fig.add_trace(go.Scatter(x=x_lo, y=y_lo, mode='lines',
                                     line=dict(color=outline_color, width=1, dash='dot'),
                                     showlegend=False, hoverinfo='skip',
                                     legendgroup=legendgroup))
            fig.add_trace(go.Scatter(x=x_up, y=y_up, mode='lines',
                                     line=dict(color=outline_color, width=1, dash='dot'),
                                     showlegend=False, hoverinfo='skip',
                                     legendgroup=legendgroup))
            i = j


class ParityPlot(IGBasicPlot, AnalysisMixin):

    def _populate_plot(self, x: np.ndarray, y: np.ndarray, **kwargs) -> None:
        _range = [min(x), max(x)]
        nbins = int(kwargs.get("nbins", 100))

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
        self.add_step_polygon(
            self._fig,
            edges=edges, lower=lq_e, upper=uq_e,
            outline_color=self._accent_1_opaque,
            fill_color=self._accent_1_opaque_fill,
            legendgroup="median"
        )

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


class BiasPlot(IGBasicPlot, AnalysisMixin):

    def _populate_plot(self, x: np.ndarray, y: np.ndarray, **kwargs) -> None:
        _range = [min(x), max(x)]
        nbins = int(kwargs.get("nbins", 100))

        y_edges = np.linspace(max(-max(x), min(y)), min(max(x), max(y)), nbins + 1)
        x_edges = np.linspace(min(x), max(x), nbins + 1)
        H, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])

        Z = H.astype(int)

        y_centers = (y_edges[:-1] + y_edges[1:]) / 2
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
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
            x=x_centers, y=y_centers, z=Z.T,
            zauto=False, zmin=0,
            coloraxis="coloraxis",
            showlegend=True, name="Data",
            hoverongaps=False
        ))

        # calculate binned median and containment
        edges, m_e, lq_e, uq_e = self.get_median_and_containment(x, y, int(nbins / 2))

        # plot containment
        self.add_step_polygon(
            self._fig,
            edges=edges, lower=lq_e, upper=uq_e,
            outline_color=self._accent_1_opaque,
            fill_color=self._accent_1_opaque_fill,
            legendgroup="median"
        )

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


class ConfusionMatrixPlot(IGBasicPlot):
    def _populate_plot(self, x: np.ndarray, y: np.ndarray, **kwargs) -> None:
        y_pred = x.argmax(axis=1).astype(int)
        y_true = y.astype(int)

        # infer number of classes
        n_classes = int(max(y_pred.max(initial=0), y_true.max(initial=0))) + 1
        labels = [str(i) for i in range(n_classes)]

        # counts
        cm = np.zeros((n_classes, n_classes), dtype=np.int64)
        np.add.at(cm, (y_true, y_pred), 1)

        # use log10(count + 1) so zeros stay finite
        z = np.log10(cm.astype(float) + 1.0)
        zmax = float(z.max()) if z.size else 1.0

        # colorbar ticks
        max_count = int(cm.max()) if cm.size else 0
        if max_count <= 1:
            bases = [0, 1]
        else:
            top_k = math.floor(math.log10(max_count))
            bases = [v for k in range(top_k + 1) if (v := 10 ** k) <= max_count]
        tickvals = [np.log10(b + 1.0) for b in bases]
        ticktext = [str(b) for b in bases]

        hover_text = np.empty(cm.shape, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                hover_text[i, j] = (
                    f"Pred: {labels[j]}<br>"
                    f"True: {labels[i]}<br>"
                    f"Count: {cm[i, j]}"
                )

        # heatmap
        self._fig.add_trace(go.Heatmap(
            z=z,
            x=labels,
            y=labels,
            coloraxis="coloraxis",
            text=hover_text,
            hoverinfo="text",
        ))

        self._fig.update_layout(
            coloraxis=dict(
                colorscale=self._colorbar,
                cmin=0.0,
                cmax=zmax if zmax > 0 else 1.0,
                colorbar=dict(
                    title="Count (log)",
                    tickmode="array",
                    tickvals=tickvals,
                    ticktext=ticktext,
                ),
            ),
        )

        annotations = []
        for i in range(n_classes):
            for j in range(n_classes):
                annotations.append(dict(
                    x=labels[j],
                    y=labels[i],
                    text=f"{cm[i, j] / len(y_true):.4f}",
                    showarrow=False,
                    xanchor="center",
                    yanchor="middle",
                    font=dict(size=12, color="black"),
                    align="center",
                    bgcolor=self._legend_background_color,
                    bordercolor=self._legend_border_color,
                    borderwidth=1,
                    borderpad=2,
                    opacity=1.0,
                ))

        # attach to figure
        self._fig.update_layout(annotations=annotations)

        # Put (0,0) at top-left
        self._fig.update_yaxes(autorange="reversed")


class ROCPlot(IGBasicPlot):

    def _populate_plot(self, x: np.ndarray, y: np.ndarray, **kwargs) -> None:
        y_pred = x.astype(float)
        y_true = y.astype(int)

        N, C = y_pred.shape

        if C == 2:
            scores = y_pred[:, 1].astype(float)
            y_bin = (y_true == 1).astype(int)
            fpr, tpr, auc = self._roc_curve_binary(scores, y_bin)
            self._fig.add_trace(go.Scatter(
                x=fpr, y=tpr, mode='lines',
                line=dict(color=self._accent_1, width=3),
                fill="tozeroy", fillcolor=self._accent_1_opaque_fill,
                name=f"C1 Logistic (AUC = {auc:.3f})"
            ))

        else:
            for c in range(C):
                scores_c = y_pred[:, c].astype(float)
                y_bin = (y_true == c).astype(int)
                fpr, tpr, auc = self._roc_curve_binary(scores_c, y_bin)
                self._fig.add_trace(go.Scatter(
                    x=fpr, y=tpr, mode="lines",
                    line=dict(width=3),
                    name=f"C{c} Logistic (AUC = {auc:.3f})"))

        self._fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(color=self._dark_gray, dash='dash', width=2), name="No Skill",
            showlegend=True
        ))

        self._fig.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            legend=dict(
                x=0.98, y=0.02, xanchor="right", yanchor="bottom",
                bgcolor=self._legend_background_color, bordercolor=self._legend_border_color,
                borderwidth=1, orientation="v"
            ),
            yaxis=dict(range=[0, 1])
        )

    @staticmethod
    def _roc_curve_binary(scores: np.ndarray, y_bin: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Binary ROC for given scores and targets."""
        scores = np.asarray(scores, dtype=float)
        y_bin = np.asarray(y_bin, dtype=int)

        pos_count = int(y_bin.sum())
        neg_count = int(len(y_bin) - pos_count)

        if pos_count == 0 or neg_count == 0:
            # degenerate, all one class
            return np.array([0.0, 1.0]), np.array([0.0, 1.0]), float("nan")

        order = np.argsort(-scores, kind="mergesort")
        scores = scores[order]
        y_bin = y_bin[order]

        true_pos_cum = np.cumsum(y_bin)
        false_pos_cum = np.cumsum(1 - y_bin)

        # indices of last occurrence for each unique score
        change_idx = np.flatnonzero(np.r_[scores[1:] != scores[:-1], True])
        tp = true_pos_cum[change_idx]
        fp = false_pos_cum[change_idx]

        true_pos_rate = tp / pos_count
        false_pos_rate = fp / neg_count

        # add endpoints
        false_pos_rate = np.r_[0.0, false_pos_rate, 1.0]
        true_pos_rate = np.r_[0.0, true_pos_rate, 1.0]

        auc = float(np.trapz(true_pos_rate, false_pos_rate))
        return false_pos_rate, true_pos_rate, auc
