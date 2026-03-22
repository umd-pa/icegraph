# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from pydantic import BaseModel, model_validator

__all__ = ["Config"]


class Config(BaseModel):
    lr:             float
    betas:          tuple[float, float] = (0.9, 0.999)
    eps:            float               = 1e-8
    weight_decay:   float               = 1e-2
    amsgrad:        bool                = False
    maximize:       bool                = False
