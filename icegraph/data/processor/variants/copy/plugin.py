# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import ClassVar, Any

from icegraph.data.processor import Processor
from icegraph.data.shared.profile import profile_stage
from icegraph.data.types import Envelope

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

    @profile_stage()
    def _process(self, env: Envelope) -> Envelope | None:
        self._ensure_selected(env)
        main = env.tmp[env.active]

        env.merge(
            main[env.resolve_cols(self.config.cols)],
            to=self.config.to,
            on=env.resolve_cols(self.config.by)
        )

        return env
