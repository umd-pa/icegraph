# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from abc import abstractmethod
from typing import TypeVar

from ..stage import Stage
from ..types import Envelope

__all__ = ["Processor"]


C = TypeVar("C")

class Processor(Stage[C]):
    """Base class for streaming DataFrame processors."""

    def _ensure_selected(self, env: Envelope) -> None:
        if env.active is None:
            raise RuntimeError(f"{type(self).name}: no dataframe has been selected (env.active == None).")

    @abstractmethod
    def _process(self, env: Envelope) -> Envelope | None:
        ...
