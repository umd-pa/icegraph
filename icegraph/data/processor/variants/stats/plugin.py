# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from icegraph.data.processor import Processor
from icegraph.data.envelope import Envelope
from icegraph.statistics import StatisticService
from icegraph.statistics.types import StatisticBundleStruct

from .config import StatsConfig

__all__ = ["Stats"]


class Stats(Processor[StatsConfig]):
    name: ClassVar[str] = "stats"
    version: ClassVar[int] = 1

    def build(self) -> None:
        return

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> StatsConfig:
        return StatsConfig(**config)

    def _process(self, env: Envelope) -> Envelope | None:
        self._ensure_selected(env)
        main = env.tmp[env.active]

        # grab splitmap from envelope
        splitmap = env.get_local_attr("splitmap", None)
        if splitmap is None:
            raise RuntimeError("No splitmap found. Must assign splits before computing statistics.")

        # double check splitmap is correct shape for current table
        if len(splitmap) != len(main):
            raise RuntimeError(f"Length of splitmap does not match active table ({env.active}) row count.")

        # load from config
        cols = env.resolve_cols(self.config.cols)
        stats = self.config.stats

        # build stats
        payload: dict[str | int, dict[str, StatisticBundleStruct]] = {}
        for col in cols:
            payload[col] = {}
            lut = main[col]

            unique_splits = np.unique(splitmap)

            for split in unique_splits:
                mask = (splitmap == split)
                split_lut = lut[mask]

                array = np.vstack(split_lut.to_numpy()).astype(np.float64, copy=False)

                service = StatisticService(stats)
                service.compute_from_array(array)

                payload[col][str(split)] = service.to_struct()

        # register stats
        env.set_local_attr("stats", payload)
        return env
