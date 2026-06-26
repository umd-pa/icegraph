# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Any
from pathlib import Path

from pydantic import BaseModel

__all__ = ["Config"]


class Config(BaseModel):
    # stage config
    extractor:  StageConfig
    processors: list[StageConfig]
    writer:     StageConfig


class StageConfig(BaseModel):
    name: str
    kwargs: dict[str, Any]  # validated by each stage
