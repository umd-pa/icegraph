# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from abc import abstractmethod
from typing import Sequence, cast, TypeVar

import torch
from torch import Tensor
from torch.nn import Module, ModuleList, ModuleDict
import torch_geometric

from icegraph.common.tensors import SegmentedTensor
from icegraph.engine.components.model import Model

__all__ = ["BlockModel"]

import logging
logger = logging.getLogger(__name__)


C = TypeVar("C")


class BlockModel(Model[C]):

    # make the type checker happy
    _blocks:        ModuleList
    _activation:    Module
    _out:           Module

    def on_attach(self) -> None:
        # eagerly build out, blocks, and activation
        self._out           = self._build_out(self.in_channels, self.out_channels)
        self._activation    = self._build_activation()

        # build as module list
        self._blocks = ModuleList(self._build_blocks(self.in_channels, self.out_channels))

    def forward_pass(self, t: SegmentedTensor, /, batch: Tensor | None) -> Tensor:
        # we dont care about segments here
        data = t.data

        if batch is None:
            logger.warning("no batch was provided to the model, assuming all are nodes of the same graph")
            # if batch is not provided (single-sample inference), create a dummy batch tensor
            # this assumes all nodes belong to a single graph
            batch = torch.zeros(data.size(0), dtype=torch.long, device=data.device)

        # forward pass through each block
        for block in self.blocks:
            data = self._forward_block(data, batch, block, self.activation)

        data = torch_geometric.nn.global_mean_pool(data, batch)
        return self.out(data)

    @property
    def out(self) -> Module:
        return self._out

    @property
    def blocks(self) -> Sequence[ModuleDict]:
        return cast(Sequence[ModuleDict], cast(object, self._blocks))

    @property
    def activation(self) -> Module:
        return self._activation

    @abstractmethod
    def _build_blocks(self, in_channels: int, out_channels: int) -> list[ModuleDict]:
        ...

    @abstractmethod
    def _build_activation(self) -> Module:
        ...

    @abstractmethod
    def _build_out(self, in_channels: int, out_channels: int) -> Module:
        ...

    @abstractmethod
    def _forward_block(self, tensor: Tensor, batch: Tensor, block: ModuleDict, activation: Module) -> Tensor:
        ...
