# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import Any

from icegraph.types.factory import PluginFactory

from .service import Service

# import each built-in service
from .data import DataService
from .metrics import MetricService
from .state import StateService
from .strategy import StrategyService

__all__ = ["ServiceFactory"]


class ServiceFactory(PluginFactory[Service[Any, Any]]):
    pass


for service in [DataService, MetricService, StateService, StrategyService]:
    ServiceFactory.register(service)
