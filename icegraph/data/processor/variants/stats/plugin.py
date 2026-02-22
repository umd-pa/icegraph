# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from icegraph.data.processor import Processor
from icegraph.data.types import Envelope
from icegraph.data.shared.profile import profile_stage
from icegraph.statistics import StatisticService
from icegraph.types.statistics import  StatisticBundleStruct
from icegraph.types.data import AttributeDomain

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

    @profile_stage()
    def _process(self, env: Envelope) -> Envelope | None:
        self._ensure_selected(env)
        main = env.tmp[env.active]

        # grab splitmap from envelope
        try:
            splitmap = env.attrs[AttributeDomain.LOCAL.name]["splitmap"]
        except KeyError:
            raise RuntimeError("No splitmap found. Must assign splits before computing statistics.")

        # double check splitmap is correct shape for current table
        if len(splitmap) != len(main):
            raise RuntimeError(f"Length of splitmap does not match active table ({env.active}) row count.")

        # load from config
        cols = env.resolve_cols(self.config.cols)
        stats = self.config.stats

        # build stats
        payload: dict[str, dict[str, StatisticBundleStruct]] = {}
        for col in cols:
            payload[col] = {}
            lut = main[col]

            # load column names from compressed data
            try:
                names = env.state["compressed"][env.active][col]
            except KeyError:
                names = None

            for split in np.unique(splitmap):
                # filter the df rowwise by split
                split_lut = lut[splitmap == split]

                # stack all arrays to compute file-wide stats
                try:
                    array = np.vstack(split_lut.to_numpy()).astype(np.float64, copy=False)
                except (TypeError, ValueError) as e:
                    raise RuntimeError(
                        f"Non-numeric or non-stackable arrays in column '{col}' for split={int(split)}") from e

                # compute using the stat service and store struct
                service = StatisticService(stats, names if names is not None else range(array.shape[1]))
                service.compute_from_array(array)
                payload[col][str(split)] = service.to_struct()

        # register stats
        env.attrs[AttributeDomain.LOCAL.name]["stats"] = payload
        return env
