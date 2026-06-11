# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import Any

from icegraph.common.factory import PluginFactory

from .service import Service

# import each built-in service
from .data import DataService
from .metrics import MetricService
from .state import StateService
from .record import RecordService
from .decode import DecodeService

__all__ = ["ServiceFactory"]


class ServiceFactory(PluginFactory[Service[Any]]):
    pass


# want to register each explicitly
for service in [DataService, MetricService, StateService, RecordService, DecodeService]:
    ServiceFactory.register(service)
