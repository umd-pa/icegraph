# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

__all__ = ["EdgeBuilderConfig", "EdgeWeightSelection"]


class EdgeWeightSelection(BaseModel):
    name: str
    kwargs: dict[str, Any]  # this can vary by weight so not enforced here


class EdgeBuilderConfig(BaseModel):
    # weight modules run on the edge index, their outputs are concatenated in this order
    weights: list[EdgeWeightSelection] = Field(default_factory=list)
