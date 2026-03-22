# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import ClassVar, Any

from icegraph.data.processor import Processor
from icegraph.data.types import Envelope

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

    def _process(self, env: Envelope) -> Envelope | None:
        self._ensure_selected(env)
        main = env.tmp[env.active]

        # load rename map or cols/out
        map_ = self.config.map_

        cols = env.resolve_cols(self.config.cols)
        out = env.resolve_cols(self.config.out)

        if len(cols) != len(out):
            raise RuntimeError("Renamer: resolved 'cols' and 'out' must have the same length.")

        if not map_:
            map_ = dict(zip(cols, out))

        # ensure no missing keys
        missing = map_.keys() - set(main.columns)
        if missing:
            raise KeyError(f"Columns not found in frame: {missing}")

        # rename and return
        env.tmp[env.active] = main.rename(columns=map_)
        return env
