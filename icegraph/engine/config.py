# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Generic, TypeVar

from pydantic import BaseModel

from .components.config import _ComponentGroup
from .services.config import _ServiceGroup
from .policy.config import PolicyConfig

__all__ = ["EngineConfig"]


_SG = TypeVar("_SG", bound=_ServiceGroup)
_CG = TypeVar("_CG", bound=_ComponentGroup)


class EngineConfig(BaseModel, Generic[_SG, _CG]):
    # engine config
    services:   _SG
    components: _CG

    # static shape
    policy: PolicyConfig | None

    # debug mode
    debug: bool = False
