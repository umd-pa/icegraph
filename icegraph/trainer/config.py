# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Any
from typing_extensions import TypedDict  # python < 3.12

from pydantic import BaseModel, DirectoryPath

__all__ = ["TrainerConfig", "ComponentOption"]


class TrainerConfig(BaseModel):
    # trainer config
    seed:           int
    max_epochs:     int
    val_interval:   int

    # paths
    outdir:     DirectoryPath

    # training policy
    policy:     str

    # engine config
    services:   ServicesConfig
    components: ComponentConfig

    # debug mode
    debug: bool = False


class ServicesConfig(TypedDict):
    # these are required services
    state:      dict[str, Any]  # validated downstream
    record:     dict[str, Any]  # validated downstream
    data:       dict[str, Any]  # validated downstream
    metrics:    dict[str, Any]  # validated downstream
    decode:     dict[str, Any]  # validated downstream


class ComponentConfig(BaseModel):
    model:          ComponentOption
    normalizer:     ComponentOption
    optimizer:      ComponentOption
    loss:           ComponentOption
    transformer:    ComponentOption
    adapter:        FixedComponentOption


class FixedComponentOption(BaseModel):
    kwargs: dict[str, Any]


class ComponentOption(BaseModel):
    name: str
    kwargs: dict[str, Any]  # this is going to be validated by the components themselves
