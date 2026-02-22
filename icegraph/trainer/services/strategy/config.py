# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

__all__ = ["StrategyConfig"]


class StrategyConfig(BaseModel):
    strategy: PluginConfig


class PluginConfig(BaseModel):
    name:   str
    kwargs: dict[str, Any]
