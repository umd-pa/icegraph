# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Any
from typing_extensions import TypedDict  # python < 3.12

from pydantic import BaseModel, DirectoryPath, ConfigDict

__all__ = ["InferenceConfig", "ComponentOption"]


class InferenceConfig(BaseModel):
    # paths
    outdir:     DirectoryPath

    # model policy
    policy:     str

    services:   ServicesConfig
    components: ComponentConfig

    # debugging
    debug:      bool = False


class ServicesConfig(TypedDict):
    # these are required services
    state:   dict[str, Any]  # validated downstream
    record:  dict[str, Any]  # validated downstream
    decode:  dict[str, Any]


class ComponentConfig(BaseModel):
    # ignore any extra component configs, only want these ones
    model_config = ConfigDict(extra="ignore")

    model:          ComponentOption
    normalizer:     ComponentOption
    transformer:    ComponentOption
    adapter:        FixedComponentOption


class FixedComponentOption(BaseModel):
    kwargs: dict[str, Any]


class ComponentOption(BaseModel):
    name: str
    kwargs: dict[str, Any]  # this is going to be validated by the components themselves
