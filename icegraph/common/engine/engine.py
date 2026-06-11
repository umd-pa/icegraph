# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from abc import abstractmethod, ABC

from icegraph.common.plugins import Plugin

__all__ = ["Engine"]


class Engine(ABC):

    # engine must provide compatibility hook
    @abstractmethod
    def ensure_compatible(self, p: Plugin, /) -> None:
        ...
