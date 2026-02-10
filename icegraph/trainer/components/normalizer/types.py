# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from dataclasses import dataclass

from ..types import ComponentContext

__all__ = ["NormalizerContext"]


@dataclass(frozen=True)
class NormalizerContext(ComponentContext):
    pass
