# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import TypeVar, final
from abc import abstractmethod, ABC

import torch
from torch import Tensor

from icegraph.common.tensors import SegmentedTensor

from ..component import Component

__all__ = ["EdgeBuilder"]


C = TypeVar("C")


class EdgeBuilder(Component[C], ABC):
    """Builds graph connectivity on the accelerator from node feature columns.

    Positions are taken from the raw feature block, so this must run before the transformer and the
    normalizer, both rescale columns independently and would distort the metric
    the neighbour search operates in.
    """

    @final
    @torch.no_grad()
    def forward(self, t: SegmentedTensor, /, batch: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass through the edge builder."""
        edge_index = self.build_index(t, batch)

        # internal validation
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(
                f"Edge builder must produce an edge index of shape [2, E], "
                f"got shape {tuple(edge_index.shape)}."
            )

        if edge_index.dtype != torch.long:
            raise ValueError(
                f"Edge builder must produce an edge index of dtype long, "
                f"got {edge_index.dtype}."
            )

        edge_attr = self.build_attr(t, edge_index)

        # internal validation
        if edge_attr.ndim != 2 or edge_attr.shape[0] != edge_index.shape[1]:
            raise ValueError(
                f"Edge builder must produce edge attributes of shape [E, ATTR] with "
                f"E={edge_index.shape[1]}, got shape {tuple(edge_attr.shape)}."
            )

        # run contract validator
        self._run_forward_validator(edge_index)

        return edge_index, edge_attr

    @abstractmethod
    def build_index(self, t: SegmentedTensor, batch: Tensor) -> Tensor:
        ...

    @abstractmethod
    def build_attr(self, t: SegmentedTensor, edge_index: Tensor) -> Tensor:
        ...
