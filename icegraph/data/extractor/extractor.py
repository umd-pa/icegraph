# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from abc import abstractmethod
from typing import ClassVar, TypeVar
from pathlib import Path

from ..stage import Stage
from ..types import Envelope

__all__ = ["Extractor"]


C = TypeVar("C")

class Extractor(Stage[C]):
    """Base class for streaming data extractors."""
    file_ext: ClassVar[str]

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()

        if getattr(cls, "file_ext", None) is None:
            raise RuntimeError(f"Extractor '{cls.__name__}' must implement the class variable 'file_ext'.")

    @abstractmethod
    def _process(self, infile: Path) -> Envelope | None:
        ...
