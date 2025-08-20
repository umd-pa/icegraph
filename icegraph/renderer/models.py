# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from pathlib import Path
from typing import Optional

import numpy as np
import plotly.graph_objects as go

from .base import IGPlot, IGDistributionPlot
from icegraph.data.processor import generate_vector_mapping
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

__all__ = ["FeaturePlot", "CDFPlot", "PDFPlot", "ChargeDistributionPlot"]


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


class FeaturePlot(IGPlot):
    """
    A plotting utility for visualizing specific DOM-level features.
    """

    subplots = (1, 3)

    def plot(self, feature: str, save_path: Path | None=None) -> None:
        """
        Generate a plot of the specified feature across all DOMs for all events.

        Args:
            feature (str): Name of the DOM-level feature.
            save_path (Path): Path to save the plot. Defaults to output_dir specified in the global config file.
        """
        if not save_path:
            save_path = Path(self._config.user_config.io.default_dir) / f"feature_plot_{feature}.png"

        # label axes
        self._ax[0].set_xlabel("dom_x")
        self._ax[1].set_xlabel("dom_y")
        self._ax[2].set_xlabel("dom_z")

        self._ax[0].set_title(f"Feature {feature} vs DOM X Position")
        self._ax[1].set_title(f"Feature {feature} vs DOM Y Position")
        self._ax[2].set_title(f"Feature {feature} vs DOM Z Position")

        # determine the vector index of the feature to plot
        inverted_vector_map: dict[str, int] = generate_vector_mapping(self._config, invert=True)

        try:
            feature_idx = inverted_vector_map[feature]
        except KeyError:
            raise KeyError(f"Invalid feature '{feature}', please select from {list(inverted_vector_map.keys())}.")

        # collect all datasets to pull features from
        datasets = [
            self._registry.train_dataset,
            self._registry.test_dataset,
            self._registry.val_dataset
        ]

        # pull features from data
        array_stack = []
        for dataset in datasets:
            for idx in range(len(dataset)):
                features, labels, _, _ = dataset.get(idx)

                feature_data = features[:, feature_idx:feature_idx + 1]
                dom_coords = features[:, len(inverted_vector_map):len(inverted_vector_map) + 3]

                array_stack.append(np.hstack((feature_data, dom_coords)))

        data = np.vstack(array_stack)

        self._ax[0].hist(data[:, 1], bins=40, weights=data[:, 0])
        self._ax[1].hist(data[:, 2], bins=40, weights=data[:, 0])
        self._ax[2].hist(data[:, 3], bins=40, weights=data[:, 0])

        self.save(save_path)
