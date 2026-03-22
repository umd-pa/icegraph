# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import ClassVar, Any

import pandas as pd

from icegraph.data.processor import Processor
from icegraph.data.types import Envelope

from .config import MapConfig

__all__ = ["Mapper"]


class Mapper(Processor[MapConfig]):
    """Map values of a column to new values."""
    name: ClassVar[str] = "map"
    version: ClassVar[int] = 1

    def build(self) -> None:
        return

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> MapConfig:
        return MapConfig(**config)

    def _process(self, env: Envelope) -> Envelope | None:
        self._ensure_selected(env)
        main = env.tmp[env.active]

        # load config values
        col = self.config.col
        out = self.config.out or col
        map_ = self.config.map_

        # ensure col is valid
        if col not in main.columns:
            raise RuntimeError(f"Missing column '{col}' in active table '{env.active}'")

        # ensure col is valid
        if out in main.columns and out != col:
            raise RuntimeError(f"Output column '{out}' already exists in active table '{env.active}'")

        # get unique set of values in col
        values = pd.unique(main[col])

        if self.config.strict:
            # forbid strict mode if nans exist
            if pd.isna(values).any():
                raise RuntimeError(f"Column '{col}' contains NaN; cannot map in strict mode.")

            # check for unmapped values
            missing = [v for v in values if v not in map_]
            if missing:
                raise RuntimeError(f"All values must be mapped if strict is True. Unmapped values: {missing}")

        mapped = main[col].map(map_)
        main[out] = mapped if self.config.strict else mapped.fillna(main[col])

        return env
