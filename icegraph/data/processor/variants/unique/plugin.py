# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import ClassVar, Any

from icegraph.data.processor import Processor
from icegraph.data.envelope import Envelope
from icegraph.common.data import AttributeDomain

from .config import UniqueConfig

__all__ = ["Unique"]

import logging
logger = logging.getLogger(__name__)


class Unique(Processor[UniqueConfig]):
    """Determine all unique values in each given column."""
    name: ClassVar[str] = "unique"
    version: ClassVar[int] = 1

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> UniqueConfig:
        return UniqueConfig(**config)

    def build(self) -> None:
        return

    def _process(self, item: Envelope) -> Envelope | None:
        active = self._require_active(item)
        main = item.tmp[active]

        # grab from config
        cols = item.resolve_cols(self.config.cols)

        # update for each col
        for col in cols:
            item.set_column_attr(
                col, "unique", main[col].dropna().unique().tolist(), domain=AttributeDomain.LOCAL
            )

        return item
