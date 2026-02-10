# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

__all__ = ["MetricConfig"]


class MetricConfig(BaseModel):
    select: list[MetricSelection]


class MetricSelection(BaseModel):
    name: str
    kwargs: dict[str, Any]  # this can vary by metric so not enforced here
