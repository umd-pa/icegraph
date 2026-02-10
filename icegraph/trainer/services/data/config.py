# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

__all__ = ["DataConfig"]


class DataConfig(BaseModel):
    module: ModuleConfig
    reader: ReaderConfig
    loader: LoaderConfig


class ModuleConfig(BaseModel):
    targets:    list[str]
    aux:        list[str]


class ReaderConfig(BaseModel):
    name: str


class LoaderConfig(BaseModel):
    batch_size:         int
    block_size:         int
    num_workers:        int
    prefetch_factor:    int
    mp_context:         Literal["fork", "spawn", "forkserver"]
    persistent_workers: int
