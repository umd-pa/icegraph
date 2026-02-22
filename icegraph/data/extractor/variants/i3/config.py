# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Any
from pathlib import Path

from pydantic import BaseModel, FilePath

__all__ = ["I3ExtractorConfig"]


class I3ExtractorConfig(BaseModel):
    gcd_path:   FilePath
    include:    list[str]
    ml_suite:   dict[str, Any]  # validation is up to ml_suite
    mclabeler:  dict[str, Any]  # validation is up to mclabeler
