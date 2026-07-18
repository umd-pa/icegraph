# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import ClassVar, Any

from icegraph.data.processor import Processor
from icegraph.data.envelope import Envelope

from .config import CommitConfig

__all__ = ["Committer"]


class Committer(Processor[CommitConfig]):
    """Load a frame into tmp for processing."""
    name: ClassVar[str] = "commit"
    version: ClassVar[int] = 1

    def build(self) -> None:
        return

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> CommitConfig:
        return CommitConfig(**config)

    def _process(self, item: Envelope) -> Envelope | None:
        active = self._require_active(item)
        main = item.tmp[active]

        # load from config
        ids = item.resolve_cols(self.config.ids)
        cols = item.resolve_cols(self.config.cols)

        # ensure properly compressed before commit
        if main.select(ids).is_duplicated().any():
            raise RuntimeError(f"Columns {cols} must form a unique key. Did you compress?")

        # commit each col one at a time so attrs can be set for each
        for col in cols:
            item.commit(main.select(ids + [col]), on=ids, validate="1:1")

        return item
