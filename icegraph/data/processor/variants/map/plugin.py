# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import ClassVar, Any

from icegraph.data.processor import Processor
from icegraph.data.envelope import Envelope

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

    def _process(self, item: Envelope) -> Envelope | None:
        active = self._require_active(item)
        main = item.tmp[active]

        # load config values
        col = str(self.config.col)
        out = str(self.config.out) if self.config.out is not None else col
        map_ = self.config.map_

        # ensure col is valid
        if col not in main.columns:
            raise RuntimeError(f"Missing column '{col}' in active table '{item.active}'")

        # ensure col is valid
        if out in main.columns and out != col:
            raise RuntimeError(f"Output column '{out}' already exists in active table '{item.active}'")

        series = main.get_column(col)

        if self.config.strict:
            # get unique set of values in col
            unique = series.unique()

            # forbid strict mode if nulls/nans exist
            if unique.null_count() > 0 or (unique.dtype.is_float() and unique.is_nan().any()):
                raise RuntimeError(f"Column '{col}' contains NaN; cannot map in strict mode.")

            # check for unmapped values
            missing = [v for v in unique.to_list() if v not in map_]
            if missing:
                raise RuntimeError(f"All values must be mapped if strict is True. Unmapped values: {missing}")

            mapped = series.replace_strict(map_)
        else:
            # unmapped values keep their original value
            mapped = series.replace(map_)

        item.tmp[active] = main.with_columns(mapped.alias(out))

        return item
