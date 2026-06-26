# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from pydantic import BaseModel

__all__ = ["GravNetConfig"]


class GravNetConfig(BaseModel):
    hidden_layers:          int
    hidden_channels:        int
    num_neighbors:          int
    space_dimensions:       int
    propagate_dimensions:   int
