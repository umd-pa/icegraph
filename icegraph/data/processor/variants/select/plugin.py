# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import ClassVar, Any

import pyarrow as pa
import pandas as pd

from icegraph.data.processor import Processor
from icegraph.data.envelope import Envelope

from .config import SelectConfig

__all__ = ["Selector"]


class Selector(Processor[SelectConfig]):
    """Load a frame into tmp for processing."""
    name: ClassVar[str] = "select"
    version: ClassVar[int] = 1

    def build(self) -> None:
        return

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> SelectConfig:
        return SelectConfig(**config)

    def _process(self, item: Envelope) -> Envelope | None:
        # set active frame
        item.active = self.config.key

        if item.active not in item.tmp:
            # load required data from envelope if not in tmp yet
            table: pa.Table = item.quiver.get(item.active)
            df: pd.DataFrame = table.to_pandas()

            if df is None:
                raise RuntimeError(f"Could not resolve key '{item.active}' in data.")

            # cache to tmp
            item.tmp[item.active] = df

        return item
