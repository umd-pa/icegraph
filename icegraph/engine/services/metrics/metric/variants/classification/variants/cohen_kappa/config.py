# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

__all__ = ["CohenKappaConfig", "KappaWeights"]


KappaWeights = Literal["none", "linear", "quadratic"]


class CohenKappaConfig(BaseModel):
    weights: KappaWeights = "none"
