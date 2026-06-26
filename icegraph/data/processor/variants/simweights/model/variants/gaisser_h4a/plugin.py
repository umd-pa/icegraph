# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import ClassVar, Any
from functools import cached_property

import simweights
from numpy.typing import ArrayLike

from icegraph.data.processor.variants.simweights.model import FluxModel

from .config import GaisserH4aConfig

__all__ = ["GaisserH4a"]


class GaisserH4a(FluxModel[GaisserH4aConfig]):
    """Gaisser H4a cosmic-ray flux model."""
    name: ClassVar[str] = "gaisser-h4a"
    version: ClassVar[int] = 1

    def build(self) -> None:
        return None

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> GaisserH4aConfig:
        return GaisserH4aConfig(**config)

    @cached_property
    def _flux(self):
        return simweights.GaisserH4a()

    # need to do some Liskov violations because simweights was designed strangely
    def __call__(self, energy: ArrayLike, pdgid: ArrayLike) -> ArrayLike:  # pyright: ignore[reportIncompatibleMethodOverride]
        return self._flux(energy, pdgid)