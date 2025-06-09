# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

from icegraph.dataset import IGData


class IGPlot(ABC):

    def __init__(self, data: IGData) -> None:
        self._data: IGData = data

        # initialize figure
        self._ax: plt.Axes
        self._fig: plt.Figure
        self._fig, self._ax = plt.subplots(1, 1)

    def plot_sampled_cdf(self) -> None:
        pass
