# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from pydantic import DirectoryPath, model_validator
from typing import Self
import re

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
    run:            str = ''

    # paths
    outdir:     DirectoryPath

    @model_validator(mode="after")
    def validate_run(self) -> Self:
        if self.run != '' and not re.fullmatch(r"[A-Za-z0-9_-]+", self.run):
            raise ValueError("Run name may only contain letters, digits, '-' and '_'.")

        return self
