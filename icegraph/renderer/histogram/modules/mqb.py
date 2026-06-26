# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Iterator
from typing_extensions import override

import plotly.graph_objects as go
import numpy as np

from icegraph.common.histogram import Histogram
from icegraph.renderer.style import PLOT_STYLE
from icegraph.typing.common import ArrayF64, ArrayI64, ArrayB

from ..module import HistogramPlotterModule
from ..plotter import HistogramPlotter2D

__all__ = ["MedianQuantileBand"]


class MedianQuantileBand(HistogramPlotterModule):
    compatible = (HistogramPlotter2D,)

    @override
    def overlay(self, fig: go.Figure, data: dict[str | int, Histogram]) -> None:
        for i, (key, item) in enumerate(data.items()):
            self._overlay_trace(fig, item, str(key))

    def _overlay_trace(self, fig: go.Figure, data: Histogram, label: str) -> None:
        # get binned median and containment along axis 0
        edges = data.edges[0]
        centers = data.centers[1]

        median = self._indices_to_centers(data.count_quantile(0.50, axis=0), centers)
        low_q = self._indices_to_centers(data.count_quantile(0.32, axis=0), centers)
        high_q = self._indices_to_centers(data.count_quantile(0.68, axis=0), centers)

        # duplicate last entry to match edges
        median = np.r_[median, median[-1]]
        low_q = np.r_[low_q, low_q[-1]]
        high_q = np.r_[high_q, high_q[-1]]

        # plot containment
        self._plot_median_and_confidence(
            fig, edges=edges, lower=low_q, upper=high_q, median=median, label=label
        )

    @staticmethod
    def _indices_to_centers(indices: ArrayI64, centers: ArrayF64) -> ArrayF64:
        # init with nan
        out = np.full(indices.shape, np.nan, dtype=centers.dtype)

        # indices of -1 indicate invalid result
        valid = indices >= 0

        out[valid] = centers[indices[valid]]

        return out

    @staticmethod
    def _iter_masked_blocks(n: int, valid: ArrayB) -> Iterator[tuple[int, int]]:
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
            edges: ArrayF64,
            lower: ArrayF64,
            upper: ArrayF64,
            median: ArrayF64,
            label: str
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
                fill='toself', fillcolor=PLOT_STYLE.accent_1_opaque_fill,
                hoverinfo='skip', showlegend=False,
                legendgroup=f"mqb-{label}"
            ))

            # draw in fill outlines
            for array in [_upper, _lower]:
                fig.add_trace(go.Scatter(
                    x=_edges, y=array, mode='lines',
                    line=dict(color=PLOT_STYLE.accent_1_opaque, width=1, dash='dot'),
                    showlegend=False, hoverinfo='skip',
                    legendgroup=f"mqb-{label}"
                ))

            # plot the median line, along wit a white glow so it stands out in front
            # of the filled containment region
            fig.add_trace(go.Scatter(
                x=_edges, y=_median, mode='lines',
                line=dict(color="white", width=3, shape="hv"),
                showlegend=False, legendgroup=f"mqb-{label}",
                hoverinfo='skip'
            ))
            fig.add_trace(go.Scatter(
                x=_edges, y=_median, mode='lines',
                line=dict(color=PLOT_STYLE.accent_1, width=2, shape="hv"),
                name=f'{label} - Median w/ 68% Containment' if block == 0 else None,
                showlegend=block == 0, legendgroup=f"mqb-{label}"
            ))

            # increment block counter
            block += 1
