# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, model_validator

__all__ = ["AdamWConfig", "SGDConfig"]


class AdamWConfig(BaseModel):
    lr:             float
    betas:          tuple[float, float] = (0.9, 0.999)
    eps:            float               = 1e-8
    weight_decay:   float               = 1e-2
    amsgrad:        bool                = False
    maximize:       bool                = False


class SGDConfig(BaseModel):
    lr:             float
    momentum:       float   = 0.0
    dampening:      float   = 0.0
    weight_decay:   float   = 0.0
    nesterov:       bool    = False
    maximize:       bool    = False

    @model_validator(mode="after")
    def _check_nesterov(self) -> Self:
        if self.nesterov and self.momentum <= 0:
            raise ValueError("nesterov=True requires momentum > 0")
        return self
