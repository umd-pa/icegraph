# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

__all__ = ["StandardI3DecoderConfig", "FluxConfig"]


class FluxConfig(BaseModel):
    # which package the model comes from
    source: Literal["simweights", "nuflux"]

    # model name
    name:   str

    # constructor arguments for a simweights model, property assignments for a
    # nuflux one
    kwargs: dict[str, Any] = Field(default_factory=dict)


class StandardI3DecoderConfig(BaseModel):
    flux:           dict[str, FluxConfig] = Field(default_factory=dict)
    surface_attr:   str = "surface"
