# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING
from pathlib import Path

from icegraph.data.base.operator import Operator

if TYPE_CHECKING:
    from icegraph.data.pipeline import Pipeline
else:
    Pipeline = None


class Extractor(Operator):

    def bootstrap(self, infile: Path) -> Optional[Path]:
        """Extractors simply need an iterator over the input file paths, thus just return the path."""
        if self._parent is None:
            raise RuntimeError("Stage has no parent; set_parent(...) before bootstrap(...).")
        return infile

    @abstractmethod
    def _process(self, infile: Path) -> Optional[Pipeline.Envelope]:
        """Take envelope, return envelope with transformed data (or None to drop)."""
        ...