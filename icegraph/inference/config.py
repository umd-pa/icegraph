# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from pydantic import DirectoryPath
from pathlib import Path

from icegraph.common.engine import ComponentKind

from ..engine.components import component_group
from ..engine.services import service_group
from ..engine.config import EngineConfig

__all__ = ["InferenceConfig"]


InferenceServiceConfig = service_group(
    "state", "record", "decode", "data",
    name="InferenceServiceConfig"
)


InferenceComponentConfig = component_group(
    ComponentKind.MODEL, ComponentKind.NORMALIZER, ComponentKind.TRANSFORMER,
    name="InferenceComponentConfig"
)


class InferenceConfig(EngineConfig[InferenceServiceConfig, InferenceComponentConfig]):
    # paths
    outdir:     DirectoryPath
    model_path: Path
