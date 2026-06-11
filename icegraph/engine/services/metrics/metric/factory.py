# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from icegraph.common.factory import PluginFactory

from .metric import Metric

# implementations
from . import variants

__all__ = ["MetricFactory"]


class MetricFactory(PluginFactory[Metric]):
    pass


for name in variants.__all__:
    MetricFactory.register(getattr(variants, name))
