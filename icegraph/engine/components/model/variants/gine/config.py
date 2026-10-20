# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from pydantic import BaseModel

__all__ = ["GINEConfig"]


class GINEConfig(BaseModel):
    hidden_layers:          int
    hidden_channels:        int
    edge_channels:          int = 1
    train_eps:              bool = False
    checkpoint:             bool = False
