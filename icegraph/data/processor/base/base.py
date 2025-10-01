# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING, Type, List
from pathlib import Path

import pandas as pd

from icegraph.data.base import Stage

if TYPE_CHECKING:
    from icegraph.data.pipeline import Pipeline
else:
    Pipeline = None


class Processor(Stage):
    """Base class for streaming DataFrame processors."""

    # prerequisite flag
    PRE_REQS: Optional[List[Type[Stage]]] = None

    def bootstrap(self, infile: Path) -> Optional[Pipeline.Envelope]:
        if self._parent is None:
            raise RuntimeError("Stage has no parent; set_parent(...) before bootstrap(...).")

        # create empty df and a file handle
        df = pd.DataFrame()
        fh = self._parent.FileHandle(src=infile)

        return self._parent.Envelope(df=df, fh=fh)

    @abstractmethod
    def _process(self, env: Pipeline.Envelope) -> Optional[Pipeline.Envelope]: ...
