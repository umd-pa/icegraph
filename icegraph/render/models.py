# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from .base import IGPlot
from icegraph.geometry import Detector
from icegraph.data.converter import generate_vector_mapping

import numpy as np
from pathlib import Path


class FeaturePlot(IGPlot):
    """
    A plotting utility for visualizing specific DOM-level features.
    """

    def plot_feature(self, feature: str, event_idx: int, save_path: Path | None=None) -> None:
        """
        Generate a plot of the specified feature across all DOMs for a single selected event.

        Args:
            feature (str): Name of the DOM-level feature.
            event_idx (int): Selected event index number.
            save_path (Path): Path to save the plot.
        """
        if not save_path:
            save_path = Path(self._config.user_config.output_dir) / f"feature_plot_{feature}.png"

        # determine the vector index of the feature to plot
        inverted_vector_map: dict[str, int] = generate_vector_mapping(self._config, invert=True)
        feature_idx = inverted_vector_map[feature]

        # pull features from data
        features, labels, dom_ids = self._data.get_with_dom_id(event_idx)
        values: list[float] = list(features[:, feature_idx])

        # convert OM keys to xyz coords
        dom_coords = [self._detector.get_dom_coords(*_id) for _id in dom_ids]

        data = list(zip(dom_coords, values))
        data = np.array([list(coord) + [value] for coord, value in data])

        # add to plot
        self._ax.scatter(data[:, 2], data[:, 3])
        self.save(save_path)
