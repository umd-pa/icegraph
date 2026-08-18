# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from pydantic import BaseModel

__all__ = ["LMDBWriterConfig"]


class LMDBWriterConfig(BaseModel):
    pass
