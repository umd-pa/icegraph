# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Any
from pathlib import Path

from pydantic import BaseModel
import yaml

__all__ = ["Config"]


class Config(BaseModel):
    # seed for training determinism
    seed: int

    trainer:    TrainerConfig
    services:   dict[str, Any]  # validated downstream
    components: ComponentConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        with Path(path).open("r") as file:
            return cls(**yaml.safe_load(file))


class TrainerConfig(BaseModel):
    max_epochs:     int
    val_interval:   int
    save_interval:  int


class ComponentConfig(BaseModel):
    model:      ComponentOption
    normalizer: ComponentOption
    optimizer:  ComponentOption
    loss:       ComponentOption
    scheduler:  ComponentOption


class ComponentOption(BaseModel):
    name: str
    kwargs: dict[str, Any]  # this is going to be validated by the components themselves
