# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from abc import ABC, abstractmethod
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from icegraph.data import DatasetRegistry
from icegraph.config import IGConfig
from icegraph.console import Console
from icegraph.geometry import Detector

__all__ = ["IGPlot"]


class IGPlot(ABC):

    subplots: tuple[int, int] = (1, 1)

    def __init__(self, registry: DatasetRegistry) -> None:
        self._registry: DatasetRegistry = registry
        self._config: IGConfig = IGConfig.get()

        # init detector object to convert om keys to coords
        self._detector = Detector()

        # initialize figure
        self._fig, self._ax = plt.subplots(
            self.subplots[0], self.subplots[1],
            figsize=(4 * self.subplots[1], 4 * self.subplots[0]),
            constrained_layout=True,
            squeeze=False
        )
        self._ax = list(self._ax.flatten())

        # type hint for usability
        self._ax: list[plt.Axes, ...]
        self._fig: plt.Figure

        for ax in self._ax:
            ax.set_facecolor('#eeeeee')

    @abstractmethod
    def plot(self, feature: str, save_path: Path | None=None) -> None:
        ...

    def save(self, path: Path):
        Console.out(f"Saving feature plot: {path}")
        self._fig.savefig(path)
