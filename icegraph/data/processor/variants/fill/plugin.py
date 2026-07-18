# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import ClassVar, Any

import numpy as np
import polars as pl

from icegraph.data.processor import Processor
from icegraph.data.envelope import Envelope

from .config import FillConfig

__all__ = ["Filler"]


class Filler(Processor[FillConfig]):
    """Add a new column (or replace an existing column) with a constant value in each row."""
    name: ClassVar[str] = "fill"
    version: ClassVar[int] = 1

    def build(self) -> None:
        return

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> FillConfig:
        return FillConfig(**config)

    def _process(self, item: Envelope) -> Envelope | None:
        active = self._require_active(item)
        main = item.tmp[active]

        # load from config
        col = self.config.col
        value = self.config.value
        dtype = self.config.dtype

        # fill column with value
        item.tmp[active] = main.with_columns(
            pl.Series(col, np.full(len(main), value, dtype=np.dtype(dtype)))
        )

        return item
