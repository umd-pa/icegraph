# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean'

from __future__ import annotations

from typing import TypeAlias

__all__ = ["Columns"]


Columns: TypeAlias = str | int | list[str] | list[int]
