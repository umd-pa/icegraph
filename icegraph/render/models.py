# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from .base import IGPlot
from icegraph.data.converter import generate_vector_mapping

__all__ = ["FeaturePlot"]


class FeaturePlot(IGPlot):
    """
    A plotting utility for visualizing specific DOM-level features.
    """

    subplots = (1, 3)

    def plot_feature(self, feature: str, save_path: Path | None=None) -> None:
        """
        Generate a plot of the specified feature across all DOMs for all events.

        Args:
            feature (str): Name of the DOM-level feature.
            save_path (Path): Path to save the plot. Defaults to output_dir specified in the global config file.
        """
        if not save_path:
            save_path = Path(self._config.user_config.io.output_dir) / f"feature_plot_{feature}.png"

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
            self._registry.training_dataset,
            self._registry.test_dataset,
            self._registry.validation_dataset
        ]

        # pull features from data
        array_stack = []
        for dataset in datasets:
            for idx in range(len(dataset)):
                features, labels = dataset[idx]

                feature_data = features[:, feature_idx:feature_idx + 1]
                dom_coords = features[:, len(inverted_vector_map):len(inverted_vector_map) + 3]

                array_stack.append(np.hstack((feature_data, dom_coords)))

        data = np.vstack(array_stack)

        self._ax[0].hist(data[:, 1], bins=40, weights=data[:, 0])
        self._ax[1].hist(data[:, 2], bins=40, weights=data[:, 0])
        self._ax[2].hist(data[:, 3], bins=40, weights=data[:, 0])

        self.save(save_path)
