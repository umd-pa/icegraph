# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from ...config import WriterConfig

__all__ = ["ZarrWriterConfig"]


class ZarrWriterConfig(WriterConfig):
    chunk_size: int = 8
