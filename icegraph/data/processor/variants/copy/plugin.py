# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import ClassVar, Any

from icegraph.data.processor import Processor
from icegraph.data.envelope import Envelope

from .config import CopyConfig

__all__ = ["Copier"]


class Copier(Processor[CopyConfig]):
    """Load a frame into tmp for processing."""
    name: ClassVar[str] = "copy"
    version: ClassVar[int] = 1

    def build(self) -> None:
        return

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> CopyConfig:
        return CopyConfig(**config)

    def _process(self, item: Envelope) -> Envelope | None:
        active = self._require_active(item)
        main = item.tmp[active]

        # resolve by
        by = item.resolve_cols(self.config.by)

        item.merge(
            main[list(set(item.resolve_cols(self.config.cols) + by))],
            to=self.config.to,
            on=by
        )

        return item
