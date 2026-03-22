# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import ClassVar, Any

from icegraph.data.processor import Processor
from icegraph.data.types import Envelope

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

    def _process(self, env: Envelope) -> Envelope | None:
        # set active frame
        env.active = self.config.key

        if env.active not in env.tmp:
            # load required data from envelope if not in tmp yet
            df = env.data.get(env.active)
            if df is None:
                raise RuntimeError(f"Could not resolve key '{env.active}' in data.")

            # copy to tmp
            env.tmp[env.active] = df.copy(deep=True)

        return env
