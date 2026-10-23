# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import ClassVar, Any
import operator

import numpy as np
import polars as pl

from icegraph.data.processor import Processor
from icegraph.data.envelope import Envelope

from .config import FilterConfig

__all__ = ["Filter"]


class Filter(Processor[FilterConfig]):
    """Filter data by a column value. Drops all rows that do not meet the condition."""
    name: ClassVar[str] = "filter"
    version: ClassVar[int] = 1

    def build(self) -> None:
        return

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> FilterConfig:
        return FilterConfig(**config)

    def _process(self, item: Envelope) -> Envelope | None:
        active = self._require_active(item)
        main = item.tmp[active]

        # load from config
        col = self.config.col
        value = self.config.value
        condition = self.config.condition

        # get the operator to use
        fn = getattr(operator, condition)

        # build the masker
        mask = fn(pl.col(col), value)

        # drop filtered out columns
        item.tmp[active] = main.filter(mask)

        return item
