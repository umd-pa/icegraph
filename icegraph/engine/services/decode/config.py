# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

__all__ = ["DecodeConfig", "KeyMapConfig"]


class KeyMapConfig(BaseModel):
    truth:      str = "truth"  # both targets and auxiliary are derived from truth
    features:   str = "features"
    edge_index: str = "edge_index"
    edge_attr:  str = "edge_attr"
    simweights: str = "simweights"


class DecodeConfig(BaseModel):
    attrs:      PluginConfig
    records:    PluginConfig

    features:   list[str] = Field(default_factory=list)
    targets:    list[str] = Field(default_factory=list)
    auxiliary:  list[str] = Field(default_factory=list)

    # if the user has different names in data
    keymap:     KeyMapConfig = Field(default_factory=KeyMapConfig)


class PluginConfig(BaseModel):
    name:   str
    kwargs: dict[str, Any]
