# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import ClassVar, Any

from icegraph.data.processor import Processor
from icegraph.data.shared.profile import profile_stage
from icegraph.data.types import Envelope
from icegraph.types.data import AttributeDomain

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

    @profile_stage()
    def _process(self, env: Envelope) -> Envelope | None:
        self._ensure_selected(env)
        main = env.tmp[env.active]

        # load from config
        ids = env.resolve_cols(self.config.ids)
        cols = env.resolve_cols(self.config.cols)

        # ensure properly compressed before commit
        if main.duplicated(subset=ids).any():
            raise RuntimeError(f"Columns {cols} must form a unique key. Did you compress?")

        # commit each col one at a time so attrs can be set for each
        for col in cols:
            env.commit(main[ids + [col]], on=ids, validate="1:1")

            # if compression data is available, store in attrs
            compression_map = env.state.get("compressed", {}).get(env.active, {})
            if col in compression_map:
                env.attrs[AttributeDomain.GLOBAL.name][col] = compression_map[col]

        return env
