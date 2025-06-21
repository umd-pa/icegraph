# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from abc import ABC, abstractmethod
from pathlib import Path

import matplotlib.pyplot as plt

from icegraph.data.base import IGData
from icegraph.config import IGConfig
from icegraph.console import Console
from icegraph.geometry import Detector

__all__ = ["IGPlot"]


class IGPlot(ABC):

    def __init__(self, data: IGData, config: IGConfig) -> None:
        self._data: IGData = data
        self._config: IGConfig = config

        # init detector object to convert om keys to coords
        self._detector = Detector(self._config)

        # initialize figure
        self._ax: plt.Axes
        self._fig: plt.Figure
        self._fig, self._ax = plt.subplots(1, 1)

    def save(self, path: Path):
        Console.out(f"Saving feature plot: {path}")
        self._fig.savefig(path)
