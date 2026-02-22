# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from icegraph.types.factory import PluginFactory

from .metric import Metric

# implementations
from .variants import MSE, MAE, RMSE

__all__ = ["MetricFactory"]


class MetricFactory(PluginFactory[Metric]):
    pass


for module in [MSE, MAE, RMSE]:
    MetricFactory.register(module)
