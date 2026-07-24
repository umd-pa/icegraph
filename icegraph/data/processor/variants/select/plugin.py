# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import ClassVar, Any

from icegraph.data.processor import Processor
from icegraph.data.envelope import Envelope

from .config import SelectConfig

import logging
logger = logging.getLogger(__name__)

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
        key = self.config.key

        if key not in item.tmp:
            # load required data from the quiver if not in tmp yet
            df = item.quiver.get(key)

            logger.info(f"Loading arrow '{key}' from quiver.")

            if df is None:
                raise RuntimeError(f"Could not resolve arrow '{key}' from quiver.")

            # cache to tmp
            item.tmp[key] = df

        # set active frame
        item.active = key

        return item
