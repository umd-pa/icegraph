# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Any

from .config import FluxConfig

__all__ = ["build_flux", "expected_source"]

import logging
logger = logging.getLogger(__name__)


# the flux family each simulation type is normally weighted with
_EXPECTED_SOURCE: dict[str, str] = {
    "corsika":  "simweights",
    "nugen":    "nuflux",
}


def expected_source(simulation: str) -> str | None:
    """Package a simulation type is normally weighted from, if known."""
    return _EXPECTED_SOURCE.get(simulation)


def _simweights_flux(config: FluxConfig) -> Any:
    """Build one of the cosmic ray flux models shipped with simweights."""
    import simweights

    model = getattr(simweights, config.name, None)

    if model is None or not isinstance(model, type):
        raise ValueError(
            f"simweights {simweights.__version__} defines no flux model {config.name!r}."
        )

    try:
        return model(**config.kwargs)
    except TypeError as e:
        raise TypeError(
            f"Could not build the simweights flux {config.name!r} with kwargs "
            f"{config.kwargs}: {e}"
        ) from e


def _nuflux_flux(config: FluxConfig) -> Any:
    """Build one of the atmospheric neutrino flux models shipped with nuflux."""
    import nuflux

    try:
        flux = nuflux.makeFlux(config.name)
    except Exception as e:
        available = getattr(nuflux, "availableFluxes", None)
        raise ValueError(
            f"nuflux could not build the flux model {config.name!r}: {e}"
            + (f" Available: {sorted(available())}." if available is not None else "")
        ) from e

    # nuflux models are configured by assignment rather than by constructor
    for name, value in config.kwargs.items():
        if not hasattr(flux, name):
            raise AttributeError(
                f"The nuflux model {config.name!r} has no property {name!r}."
            )
        setattr(flux, name, value)

    return flux


def build_flux(config: FluxConfig) -> Any:
    """Build the flux model the weights are computed against."""
    match config.source:
        case "simweights":
            flux = _simweights_flux(config)
        case "nuflux":
            flux = _nuflux_flux(config)
        case _:  # pragma: no cover
            raise ValueError(f"Unknown flux source {config.source!r}.")

    logger.info("[StandardI3Decoder] Built %s flux model %s.", config.source, config.name)
    return flux
