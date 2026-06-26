# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

__all__ = ["PolicyConfig"]


class PolicyConfig(BaseModel):
    name:   str
    kwargs: dict[str, Any]
