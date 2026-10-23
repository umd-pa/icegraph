# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import ClassVar, Any

import polars as pl

from icegraph.data.processor import Processor
from icegraph.data.envelope import Envelope

from .config import PivotConfig

import logging
logger = logging.getLogger(__name__)

__all__ = ["Pivoter"]


class Pivoter(Processor[PivotConfig]):
    """Pivot from long-form to wide."""
    name: ClassVar[str] = "pivot"
    version: ClassVar[int] = 1

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> PivotConfig:
        return PivotConfig(**config)

    def build(self) -> None:
        return

    def _process(self, item: Envelope) -> Envelope | None:
        active = self._require_active(item)
        main = item.tmp[active]

        # load config
        col = self.config.col
        values = self.config.values

        # quick data check
        for key in (col, values):
            if key not in main.columns:
                raise RuntimeError(f"Missing expected column '{key}' in dataframe.")

        index_cols = item.resolve_cols(self.config.index)

        item.tmp[active] = main.pivot(
            col, index=index_cols, values=values, aggregate_function=None
        )

        return item
