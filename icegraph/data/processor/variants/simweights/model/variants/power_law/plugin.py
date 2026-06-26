# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import ClassVar, Any

from numpy.typing import ArrayLike
import numpy as np

from icegraph.data.processor.variants.simweights.model import FluxModel

from .config import PowerLawConfig

__all__ = ["PowerLaw"]


class PowerLaw(FluxModel[PowerLawConfig]):
    """Basic power-law flux model."""
    name: ClassVar[str] = "power-law"
    version: ClassVar[int] = 1

    def build(self) -> None:
        return

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> PowerLawConfig:
        return PowerLawConfig(**config)

    # need to do some Liskov violations because simweights was designed strangely
    def __call__(self, energy: ArrayLike) -> ArrayLike:  # pyright: ignore[reportIncompatibleMethodOverride]
        energy = np.asarray(energy, dtype=np.float64)
        return self.config.phi0 * (energy / self.config.e0) ** -self.config.g