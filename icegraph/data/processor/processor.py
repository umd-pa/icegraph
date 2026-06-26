# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from abc import abstractmethod
from typing import TypeVar

from ..stage import Stage
from ..envelope import Envelope

__all__ = ["Processor"]


C = TypeVar("C")

class Processor(Stage[C, Envelope]):
    """Base class for streaming DataFrame processors."""

    def _require_active(self, env: Envelope) -> str:
        if env.active is None:
            raise RuntimeError(f"{type(self).name}: no dataframe has been selected (env.active == None).")
        return env.active

    @abstractmethod
    def _process(self, item: Envelope) -> Envelope | None:
        ...
