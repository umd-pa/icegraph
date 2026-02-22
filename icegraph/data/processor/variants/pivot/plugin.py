# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import ClassVar, Any

from icegraph.data.processor import Processor
from icegraph.data.shared.profile import profile_stage
from icegraph.data.types import Envelope

from .config import PivotConfig

__all__ = ["Pivoter"]


class Pivoter(Processor[PivotConfig]):
    """Pivot from long-form to wide."""
    name: ClassVar[str] = "pivot"
    version: ClassVar[int] = 1

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> PivotConfig:
        return PivotConfig(**config)

    def build(self) -> None:
        return

    @profile_stage()
    def _process(self, env: Envelope) -> Envelope | None:
        main = env.tmp.get(env.active)

        # load config
        col = self.config.col
        values = self.config.values

        # quick data check
        for key in (col, values):
            if key not in main.columns:
                raise RuntimeError(f"Missing expected column '{key}' in dataframe.")

        # equivalent to pivot_table but faster and more stable
        keys = env.resolve_cols(self.config.index) + [col]
        env.tmp[env.active] = (
            main
            .sort_values(keys, kind="mergesort")
            .drop_duplicates(subset=keys, keep="first")
            .set_index(keys)[values]
            .unstack(col)
            .reset_index()
        )

        return env
