# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Any, ClassVar

from torch import Tensor
from torch_geometric.nn import knn_graph

from icegraph.common.tensors import SegmentedTensor

from icegraph.engine.components.edges import EdgeBuilder

from .config import KNNConfig

__all__ = ["KNNEdgeBuilder"]

# module logger
import logging
logger = logging.getLogger(__name__)


class KNNEdgeBuilder(EdgeBuilder[KNNConfig]):
    """Connect every node to its ``k`` nearest neighbours within its own graph.

    The edge attributes come from the weight modules in the config. Edges follow
    the message-passing orientation used by ``torch_geometric``: ``edge_index[0]``
    is the neighbour a message travels from and ``edge_index[1]`` is the node it
    arrives at.
    """
    name: ClassVar[str] = "knn"
    version: ClassVar[int] = 1

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> KNNConfig:
        return KNNConfig(**config)

    def build(self) -> None:
        return

    def build_index(self, t: SegmentedTensor, batch: Tensor, num_graphs: int | None = None) -> Tensor:
        # the neighbour search backend rejects a strided input
        position = t.block(self.config.neighbor_cols, contiguous=True)

        # the batch vector keeps the search inside each graph
        return knn_graph(position, k=self.config.k, batch=batch, loop=False, batch_size=num_graphs)
