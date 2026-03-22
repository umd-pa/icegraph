# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Iterator, Protocol, TYPE_CHECKING

from icegraph.types.data import ModelInputRole, Split
from icegraph.statistics import StatisticService

from ..types import ServiceView

from .types import Attributes, GlobalAttributes, SizedDataset

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from torch_geometric.data import Data


__all__ = ["DataView"]


### SURFACE

class DataView(ServiceView, Protocol):
    attrs: Iterator[Attributes]
    global_attrs: GlobalAttributes

    def set_epoch(self, epoch: int) -> None: ...
    def columns(self, role: ModelInputRole, aux: bool = False) -> list[str]: ...
    def dataloader(self, split: Split) -> DataLoader: ...
    def dataset(self, split: Split) -> SizedDataset[Data]: ...
    def stats(self, split: Split, role: ModelInputRole) -> StatisticService: ...
