# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from pydantic import BaseModel, Field

__all__ = ["StatusConfig"]


class StatusConfig(BaseModel):
    # log entries kept for inspection
    log_history:    int     = Field(default=200, ge=0)

    # updates each throughput meter keeps in memory
    window:         int     = Field(default=50, ge=2)

    # count FLOPs on the first batch of each throughput stream
    profile_flops:  bool    = True
