# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import ClassVar, Any

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

        index_cols = [str(c) for c in item.resolve_cols(self.config.index)]
        keys = index_cols + [col]

        # fast path: pivot assumes uniqueness so only run here on no dupe dfs
        try:
            item.tmp[active] = (
                main
                .pivot(index=index_cols, columns=col, values=values)
                .reset_index()
            )
        except ValueError:
            logger.warning("Fast pivot failed (likely due to duplicate keys), falling back to slower duplicate-safe version.")
            item.tmp[active] = (
                main
                .groupby(keys, sort=False, observed=True)[values]
                .first()
                .unstack(col)
                .reset_index()
            )

        return item
