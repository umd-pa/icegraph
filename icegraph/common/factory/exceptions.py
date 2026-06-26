# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import Hashable, Iterable

from icegraph.exceptions import IceGraphError


class UnknownModuleError(IceGraphError):
    """Raised when an unregistered module is requested from a factory."""

    def __init__(self, name: str, available: Iterable[Hashable]):
        available = f"[{', '.join(sorted(map(str, available)))}]"
        super().__init__(f"Module '{name}' is not registered; available: {available}")
