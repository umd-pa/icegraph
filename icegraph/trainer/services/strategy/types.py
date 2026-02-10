# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import Protocol, ClassVar

__all__ = ["CompatibleModule"]

class CompatibleModule(Protocol):
    compatible: ClassVar[tuple[str, ...]]