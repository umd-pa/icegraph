# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from icegraph.types.factory import ModuleFactory

from .service import Service

# import each service
from .data import DataService
from .metrics import MetricService
from .state import StateService
from .strategy import StrategyService

__all__ = ["ServiceFactory"]


class ServiceFactory(ModuleFactory[str, Service]):
    pass


