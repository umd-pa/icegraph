# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from pydantic import BaseModel, Field

__all__ = ["AUPRCConfig"]


class AUPRCConfig(BaseModel):
    bins:           int  = Field(default=100, ge=2)
    from_logits:    bool = True
