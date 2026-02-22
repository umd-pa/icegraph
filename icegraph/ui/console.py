# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from rich.console import Console

__all__ = ["console"]


console = Console(stderr=True, force_terminal=True)
