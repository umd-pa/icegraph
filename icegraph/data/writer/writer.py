# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from abc import abstractmethod
from typing import TypeVar

from ..stage import Stage
from ..envelope import Envelope

__all__ = ["Writer"]


C = TypeVar("C")

class Writer(Stage[C]):
    """Base class for pipeline DataFrame writers."""

    @abstractmethod
    def _process(self, env: Envelope) -> Envelope | None:
        ...
