# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import TYPE_CHECKING
from dataclasses import dataclass

from icegraph.types.data import Split, ModelInputRole
from icegraph.trainer.types import ViewSurface, AttachContext

if TYPE_CHECKING:
    from icegraph.statistics import StatisticService

__all__ = ["StatSurface", "NormalizerContext"]


class StatSurface(ViewSurface):
    def stats(self, split: Split, role: ModelInputRole) -> StatisticService: ...


@dataclass(frozen=True)
class NormalizerContext(AttachContext):
    data: StatSurface
