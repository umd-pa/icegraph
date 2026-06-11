# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel

__all__ = ["RecordConfig"]


class RecordConfig(BaseModel):
    source:             Path | list[Path]
    reader:             PluginConfig
    store:              PluginConfig
    ignore_checksum:    bool = False


class PluginConfig(BaseModel):
    name:   str
    kwargs: dict[str, Any]
