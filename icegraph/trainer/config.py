# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from pydantic import DirectoryPath

from icegraph.common.engine import ComponentKind

from ..engine.components import component_group
from ..engine.services import service_group
from ..engine.config import EngineConfig

__all__ = ["TrainerConfig"]


TrainerServiceConfig = service_group(
    "state", "record", "data", "metrics", "decode",
    name="TrainerServiceConfig"
)


TrainerComponentConfig = component_group(
    *ComponentKind.all(),
    name="TrainerComponentConfig"
)


class TrainerConfig(EngineConfig[TrainerServiceConfig, TrainerComponentConfig]):
    # trainer config
    max_epochs:     int
    val_interval:   int

    # paths
    outdir:     DirectoryPath
