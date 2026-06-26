# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import ClassVar, Any

import numpy as np

from icegraph.data.processor import Processor
from icegraph.data.envelope import Envelope

from .config import SplitMapConfig

__all__ = ["SplitMapper"]

import logging
logger = logging.getLogger(__name__)


class SplitMapper(Processor[SplitMapConfig]):
    """Assign splits using a basic rng map."""

    # splitmap is a row-aligned parallel array for the active table
    # this assumes no processor before stats has reordered or filtered rows
    # without applying the same transformation to splitmap

    name: ClassVar[str] = "splitmap"
    version: ClassVar[int] = 1

    _warning_emitted: bool
    _counter: int

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> SplitMapConfig:
        return SplitMapConfig(**config)

    def build(self) -> None:
        self._counter = 0
        self._warning_emitted = False

    def _process(self, item: Envelope) -> Envelope | None:
        active = self._require_active(item)
        main = item.tmp[active]

        rng = np.random.default_rng(self.config.seed + self._counter)
        weights = np.asarray(self.config.weights, dtype=np.float64)

        # stash the split map in attrs
        # only store as uint8 as we will never have more than 255 splits
        splits = rng.choice(self.config.range_, size=len(main), p=weights)
        item.set_local_attr("splitmap", np.asarray(splits, dtype=np.uint8))

        if not self._warning_emitted:
            logger.warning(
                "Split map computed. Row order is now assumed fixed. Any reordering or "
                "filtering of rows without updating the splitmap may corrupt split-specific statistics. "
                "This warning is shown once and will be suppressed for the rest of program execution."
            )
            self._warning_emitted = True

        # increment counter for seed
        self._counter += 1

        return item
