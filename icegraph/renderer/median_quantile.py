# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import Iterator, Tuple

import plotly.graph_objects as go
import numpy as np
import numpy.typing as npt

from icegraph.common.histogram import Histogram
from .base import Plotter

__all__ = ["MedianQuantileBand"]


class MedianQuantileBand(Plotter):

    def _plot(self, fig: go.Figure, hist: Histogram) -> None:
        # core plot logic
        self._plot_base(fig, hist)

        # optional overlays
        self._plot_overlay(fig, hist)

    def _plot_overlay(self, fig: go.Figure, hist: Histogram) -> None:
        # default: no overlays
        pass

    def set_title(self, title: str) -> None:
        self._fig.update_layout(
            title_text=title,
            title_font=dict(size=24)
        )

    def set_xlabel(self, label: str) -> None:
        self._fig.update_xaxes(
            title_text=label,
            title_font=dict(size=20)
        )

    def set_ylabel(self, label: str) -> None:
        self._fig.update_yaxes(
            title_text=label,
            title_font=dict(size=20)
        )

    def _plot_base(self, fig: go.Figure, hist: Histogram) -> None:
        # set plot bounds to data ranges
        ranges = hist.bounds

        fig.update_layout(
            xaxis_range=ranges[0],
            yaxis_range=ranges[1],
        )

        # grab centers
        centers = hist.centers

        # plot histogram as heatmap with colorbar
        fig.update_layout(  # colorbar
            coloraxis=dict(
                cmin=0, cmax=float(hist.peak_value),
                colorscale=self._colorbar,
                colorbar=dict(title="Count", len=1, y=0.5)
            )
        )
        fig.add_trace(go.Heatmap(  # histogram
            x=centers[0], y=centers[1], z=hist.histogram,
            zauto=False, zmin=0,
            coloraxis="coloraxis",
            showlegend=True, name="Data",
            hoverongaps=False
        ))

        # get binned median and containment along axis 0
        edges   = hist.edges[0]
        median  = hist.count_quantile(0.50)
        low_q   = hist.count_quantile(0.32)
        high_q  = hist.count_quantile(0.68)

        # duplicate last entry to match edges
        median  = np.r_[median, median[-1]]
        low_q   = np.r_[low_q, low_q[-1]]
        high_q  = np.r_[high_q, high_q[-1]]

        # plot containment
        self._plot_median_and_confidence(
            fig, edges=edges, lower=low_q, upper=high_q, median=median
        )

    @staticmethod
    def _iter_masked_blocks(n: int, valid: npt.NDArray[np.bool_]) -> Iterator[Tuple[int, int]]:
        if valid.shape[0] < n:
            raise ValueError("Valid mask must have length equal to or greater than n.")

        # set start to 0
        i = 0

        # iterate up to n exclusive
        while i < n:
            # skip this index if the value here is invalid
            if not valid[i]:
                i += 1
                continue

            # starting at i=j, increment until we find an invalid index
            # [i:j] then represents the index range of the block to plot
            j = i
            while j < n and valid[j]:
                j += 1

            # yield the block
            yield i, j

            # set i = j to move low index to the start of the next block
            i = j

    def _plot_median_and_confidence(
            self,
            fig: go.Figure,
            edges: npt.NDArray,
            lower: npt.NDArray,
            upper: npt.NDArray,
            median: npt.NDArray
    ) -> None:
        assert lower.size == upper.size == median.size, "Lower, upper, and median must have the same size."

        n = lower.size - 1
        valid = (np.isfinite(lower) & np.isfinite(upper))[:n]

        # iterate over valid blocks
        block = 0
        for i, j in self._iter_masked_blocks(n, valid):

            # convert batch edges to corners
            # define target = upper or lower or median
            # initially L = len(edges[i:j+1]) = len(target[i:j]) + 1
            # L' = len(np.repeat(edges[i:j+1], 2)) = 2L = 2 * len(target[i:j]) + 2
            # cut off first and last value of edges:
            # L'' = 2L' - 2 = 2 * len(target[i:j]) = len(np.repeat(target[i:j], 2)) (good)
            _edges = np.repeat(edges[i:j + 1], 2)[1: -1]

            # do np.repeat for upper/lower/median to match len(_edges) and get corners
            _upper = np.repeat(upper[i:j], 2)
            _lower = np.repeat(lower[i:j], 2)
            _median = np.repeat(median[i:j], 2)

            # plot the polygon
            # concat edges and reversed edges for x
            # concat lower and reversed upper for y
            # this results in a closed polygon that can be filled
            poly_x = np.concatenate([_edges, _edges[::-1]])
            poly_y = np.concatenate([_lower, _upper[::-1]])

            fig.add_trace(go.Scatter(
                x=poly_x, y=poly_y,
                mode='lines', line=dict(width=0),
                fill='toself', fillcolor=self._accent_1_opaque_fill,
                hoverinfo='skip', showlegend=False,
                legendgroup="median"
            ))

            # draw in fill outlines
            for array in [_upper, _lower]:
                fig.add_trace(go.Scatter(
                    x=_edges, y=array, mode='lines',
                    line=dict(color=self._accent_1_opaque, width=1, dash='dot'),
                    showlegend=False, hoverinfo='skip',
                    legendgroup="median"
                ))

            # plot the median line, along wit a white glow so it stands out in front
            # of the filled containment region
            fig.add_trace(go.Scatter(
                x=_edges, y=_median, mode='lines',
                line=dict(color="white", width=3, shape="hv"),
                showlegend=False, legendgroup="median",
                hoverinfo='skip'
            ))
            fig.add_trace(go.Scatter(
                x=_edges, y=_median, mode='lines',
                line=dict(color=self._accent_1, width=2, shape="hv"),
                name='Median w/ 68% Containment' if block == 0 else None,
                showlegend=block == 0, legendgroup="median"
            ))

            # increment block counter
            block += 1
