# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import TYPE_CHECKING
from collections.abc import Sized
from functools import cached_property

import torch
from torch.utils.data import Dataset

from icegraph.common.data import GraphData
from icegraph.common.data import Split
from icegraph.typing.common import ArrayI

if TYPE_CHECKING:
    from icegraph.engine.services import ServiceManager
    from icegraph.engine.services.record import RecordService
    from icegraph.engine.services.decode import DecodeService

__all__ = ["GraphDataset"]


class GraphDataset(Dataset[GraphData], Sized):

    def __init__(self, split: Split, services: ServiceManager) -> None:
        self.split: Split = split
        self._services: ServiceManager = services

    # will need repeated fast access to decode and record services, so just cache a reference here
    @cached_property
    def _record_service(self) -> RecordService:
        return self._services.require("record", required_by=type(self))

    @cached_property
    def _decode_service(self) -> DecodeService:
        return self._services.require("decode", required_by=type(self))

    def __getitem__(self, index: int) -> GraphData:
        # ensure int index
        if not isinstance(index, int):
            raise TypeError(f"Parameter 'index' must be of type int, got '{type(index).__name__}'")

        # normalize to key
        index = int(self.keys[index])

        # load sample
        record = self._record_service[index]

        # build data object
        data = GraphData()

        # populate
        data.features   = self._decode_service.load_features(record)
        data.targets    = self._decode_service.load_targets(record)
        data.edge_index = self._decode_service.load_edge_index(record)
        data.edge_attr  = self._decode_service.load_edge_attr(record)
        data.simweights = self._decode_service.load_simweights(record)

        # only populate auxiliary if in val or test split
        data.auxiliary = (
            self._decode_service.load_auxiliary(record)
            if self.split in Split.eval()
            else torch.empty((1, 0), dtype=torch.float32)
        )

        # set num_nodes manually since not using conventional x and y
        data.num_nodes = data.features.shape[0]

        return data

    def __len__(self) -> int:
        return len(self.keys)

    @property
    def keys(self) -> ArrayI:
        # this is cached on the decoder
        return self._decode_service.get_keys(self.split)
