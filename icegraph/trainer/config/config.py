# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from icegraph.types.store import Store

from .schema import Config

__all__ = ["ConfigStore"]


class ConfigStore(Store[Config]):
    pass
