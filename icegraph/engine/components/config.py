# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Any, cast

from pydantic import BaseModel, ConfigDict, create_model, model_validator

from icegraph.common.engine import ComponentKind

import logging
logger = logging.getLogger(__name__)

__all__ = ["component_group", "ComponentConfig"]


class ComponentConfig(BaseModel):
    name:   str
    kwargs: dict[str, Any]


class _ComponentGroup(BaseModel):
    model_config = ConfigDict(extra="ignore")

    @model_validator(mode="before")
    @classmethod
    def _warn_on_extra(cls, data):
        if isinstance(data, dict):
            extras = set(data) - set(cls.model_fields)
            if extras:
                logger.warning(
                    "%s ignoring unexpected field(s): %s",
                    cls.__name__,
                    ", ".join(sorted(extras)),
                )
        return data

    def as_mapping(self) -> dict[ComponentKind, ComponentConfig]:
        return {ComponentKind(k): v for k, v in dict(self).items()}


def component_group(*kinds: ComponentKind, name: str) -> type[_ComponentGroup]:
    fields: dict[str, Any] = {
        kind.value: (ComponentConfig, ...) for kind in kinds
    }
    model = create_model(
        name,
        __base__=_ComponentGroup,
        **fields,
    )
    return cast("type[_ComponentGroup]", model)
