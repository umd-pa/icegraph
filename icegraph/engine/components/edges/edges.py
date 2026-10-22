# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import TypeVar, final
from abc import abstractmethod, ABC

import torch
from torch import Tensor
from torch.nn import ModuleList

from icegraph.common.tensors import SegmentedTensor

from ..component import Component

from .config import EdgeBuilderConfig
from .weight import EdgeWeightFactory, EdgeWeightContext

__all__ = ["EdgeBuilder"]


C = TypeVar("C", bound=EdgeBuilderConfig)


class EdgeBuilder(Component[C], ABC):
    """Builds graph connectivity on the accelerator from node feature columns.

    A subclass supplies only the edge index. The edge attributes come from the weight
    modules named in the config. Each one runs on the same edge index, and their
    outputs are concatenated along dim 1 in config order.
    """

    weights: ModuleList

    def _internal_build_before_user(self) -> None:
        super()._internal_build_before_user()

        # build each weight module before the user build, so build() can see them
        self.weights = ModuleList()
        for selection in self.config.weights:
            # structure of config is enforced by pydantic
            weight = EdgeWeightFactory.create(selection.name, **selection.kwargs)
            weight.attach(EdgeWeightContext())

            self.weights.append(weight)

    @final
    @torch.no_grad()
    def forward(self, t: SegmentedTensor, /, batch: Tensor, num_graphs: int | None = None) -> tuple[Tensor, Tensor]:
        """Forward pass through the edge builder."""
        edge_index = self.build_index(t, batch, num_graphs)

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

        # each weight module validates its own output shape
        edge_attrs = [weight(t, edge_index) for weight in self.weights]

        if edge_attrs:
            edge_attr = torch.cat(edge_attrs, dim=1)
        else:
            # no weight modules, so each edge has zero attributes
            edge_attr = t.data.new_empty((edge_index.shape[1], 0))

        # run contract validator
        self._run_forward_validator(edge_index)

        return edge_index, edge_attr

    @abstractmethod
    def build_index(self, t: SegmentedTensor, batch: Tensor, num_graphs: int | None = None) -> Tensor:
        ...
