# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import ClassVar, Any

from icegraph.data.processor import Processor
from icegraph.data.envelope import Envelope

from .config import RenameConfig

__all__ = ["Renamer"]


class Renamer(Processor[RenameConfig]):
    """Rename columns using a mapping."""
    name: ClassVar[str] = "rename"
    version: ClassVar[int] = 1

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> RenameConfig:
        return RenameConfig(**config)

    def build(self) -> None:
        return

    def _process(self, item: Envelope) -> Envelope | None:
        active = self._require_active(item)
        main = item.tmp[active]

        # load rename map or cols/out
        # normalize to str since polars column names are always strings
        map_ = self.config.map_
        if map_ is not None:
            map_ = {str(k): str(v) for k, v in map_.items()}

        if map_ is None:
            # these should both be covered by pydantic
            assert self.config.cols is not None
            assert self.config.out is not None

            cols = item.resolve_cols(self.config.cols)
            out = item.resolve_cols(self.config.out)

            if len(cols) != len(out):
                raise RuntimeError("Renamer: resolved 'cols' and 'out' must have the same length.")

            map_ = dict(zip(cols, out))

        # ensure no missing keys
        missing = map_.keys() - set(main.columns)
        if missing:
            raise KeyError(f"Columns not found in frame: {missing}")

        # rename and return
        item.tmp[active] = main.rename(map_)

        # need to manually update column metadata keys
        item.sync_column_names(map_)

        return item
