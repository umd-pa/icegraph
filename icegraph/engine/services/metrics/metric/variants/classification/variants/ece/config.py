# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from pydantic import BaseModel, Field

__all__ = ["ECEConfig"]


class ECEConfig(BaseModel):
    bins:           int  = Field(default=15, ge=1)
    from_logits:    bool = True
