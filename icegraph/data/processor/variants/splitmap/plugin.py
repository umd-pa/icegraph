# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import ClassVar, Any

import numpy as np

from icegraph.data.processor import Processor
from icegraph.data.shared.profile import profile_stage
from icegraph.data.types import Envelope
from icegraph.types.data import AttributeDomain

from .config import SplitMapConfig

__all__ = ["SplitMapper"]


class SplitMapper(Processor[SplitMapConfig]):
    """Assign splits using a basic rng map."""
    name: ClassVar[str] = "splitmap"
    version: ClassVar[int] = 1

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> SplitMapConfig:
        return SplitMapConfig(**config)

    def build(self) -> None:
        return

    @profile_stage()
    def _process(self, env: Envelope) -> Envelope | None:
        self._ensure_selected(env)
        main = env.tmp[env.active]

        rng = np.random.default_rng(self.config.seed)
        weights = np.asarray(self.config.weights, dtype=np.float64)

        # stash the split map where it is expected in attrs
        # only store as uint8 as we will almost never have more than 255 splits
        splits = rng.choice(self.config.range_, size=len(main), p=weights)
        env.attrs[AttributeDomain.LOCAL.name]["splitmap"] = splits.astype(np.uint8)

        return env
