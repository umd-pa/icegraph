# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import ClassVar, Any

import numpy as np

from icegraph.data.processor import Processor
from icegraph.data.shared.profile import profile_stage
from icegraph.data.types import Envelope

from .config import CompressConfig

__all__ = ["Compressor"]


class Compressor(Processor[CompressConfig]):
    """Pack columns to 2d arrays."""
    name: ClassVar[str] = "compress"
    version: ClassVar[int] = 1

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> CompressConfig:
        return CompressConfig(**config)

    def build(self) -> None:
        return

    @profile_stage()
    def _process(self, env: Envelope) -> Envelope | None:
        self._ensure_selected(env)
        main = env.tmp[env.active]

        # grab from config
        by = env.resolve_cols(self.config.by)
        to = self.config.to
        out = self.config.out
        cols = env.resolve_cols(self.config.cols)

        # ensure to is not the active frame
        if to == env.active:
            raise RuntimeError("Cannot merge compressed data to it's source frame.")

        packed = (
            main.groupby(by, sort=False)[cols]
            .apply(lambda g: g.to_numpy(dtype=np.float32, copy=False))
            .rename(out)
            .reset_index()
        )

        # record the compression
        env.state["compressed"][to][out] = cols

        # merge to correct frame and return
        return env.merge(packed, to=self.config.to, on=by, validate="1:1")
