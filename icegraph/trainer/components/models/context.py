# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from dataclasses import dataclass

from ..context import AttachContext

__all__ = ["ModelContext"]


@dataclass(frozen=True)
class ModelContext(AttachContext):
    pass
