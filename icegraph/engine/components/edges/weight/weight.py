# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import TypeVar, final
from abc import abstractmethod, ABC

from torch import Tensor
from torch.nn import Module

from icegraph.common.plugins import Plugin
from icegraph.common.tensors import SegmentedTensor

from .types import EdgeWeightContext

__all__ = ["EdgeWeight"]


C = TypeVar("C")


class EdgeWeight(Plugin[C, EdgeWeightContext], Module, ABC):
    """Computes edge attributes for an edge index that an edge builder already made.

    An edge builder can hold any number of these. Each one runs on the same edge
    index, and the builder concatenates their outputs along dim 1 in config order.
    """

    def build(self) -> None:
        return

    @final
    def forward(self, t: SegmentedTensor, edge_index: Tensor) -> Tensor:
        """Forward pass through the edge weight."""
        edge_attr = self.build_attr(t, edge_index)

        # internal validation
        if edge_attr.ndim != 2 or edge_attr.shape[0] != edge_index.shape[1]:
            raise ValueError(
                f"Edge weight '{self.name}' must produce edge attributes of shape [E, ATTR] with "
                f"E={edge_index.shape[1]}, got shape {tuple(edge_attr.shape)}."
            )

        return edge_attr

    @abstractmethod
    def build_attr(self, t: SegmentedTensor, edge_index: Tensor) -> Tensor:
        ...
