# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from dataclasses import dataclass

from icegraph.common.data import Split, ColumnarRole
from icegraph.engine.services.decode import DecodeService
from icegraph.statistics import StatisticService

__all__ = ["BoundsConstructorContext"]

import logging
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BoundsConstructorContext:
    decode: DecodeService

    def get_stats(self, role: ColumnarRole) -> StatisticService:
        # only ever access training stats
        return self.decode.get_stats(Split.TRAIN, role)

    def logical_index(self, column: str, role: ColumnarRole) -> int:
        """Logical index of column in role."""
        return self.decode.get_columns(role).index(column)

    def physical_index(self, column: str, role: ColumnarRole) -> int:
        """Physical index of column in role."""
        index = self.logical_index(column, role)
        return int(self.decode.get_offsets(role)[index])
