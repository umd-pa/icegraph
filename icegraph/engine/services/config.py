# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Any, TypeAlias, cast

from pydantic import BaseModel, create_model, Field, ConfigDict

__all__ = ["service_group", "ServiceConfig"]


ServiceConfig: TypeAlias = dict[str, Any]


class _ServiceGroup(BaseModel):
    # want to allow the user to design and include their own extra services in
    # addition to the required base set
    model_config = ConfigDict(extra="allow")

    def as_mapping(self) -> dict[str, ServiceConfig]:
        return dict(self)


def service_group(*keys: str, name: str) -> type[_ServiceGroup]:
    fields: dict[str, Any] = {
        k: (ServiceConfig, Field(default_factory=dict)) for k in keys
    }
    model = create_model(
        name,
        __base__=_ServiceGroup,
        **fields
    )
    return cast("type[_ServiceGroup]", model)
