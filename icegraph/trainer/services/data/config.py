# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Literal, Any

from pydantic import BaseModel

__all__ = ["DataConfig"]


class DataConfig(BaseModel):
    sampler:    PluginConfig
    module:     PluginConfig
    store:      PluginConfig
    loader:     LoaderConfig


class PluginConfig(BaseModel):
    name:   str
    kwargs: dict[str, Any]


class LoaderConfig(BaseModel):
    batch_size:         int
    block_size:         int
    num_workers:        int
    prefetch_factor:    int
    mp_context:         Literal["fork", "spawn", "forkserver"]
    persistent_workers: int
