# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from icegraph.types.factory import ModuleFactory

from .metric import Metric

# implementations
from .standard import MSE, MAE, RMSE

__all__ = ["MetricFactory"]


class MetricFactory(ModuleFactory[str, Metric]):
    pass


for module in [MSE, MAE, RMSE]:
    MetricFactory.register(module)
