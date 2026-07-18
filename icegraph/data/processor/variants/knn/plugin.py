# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import ClassVar, Any

import numpy as np
import polars as pl
from scipy.spatial import KDTree

from icegraph.data.processor import Processor
from icegraph.data.envelope import Envelope
from icegraph.typing.common import ArrayF64, ArrayI64, ArrayI

from .config import KNNConfig

__all__ = ["KNN"]

# module logger
import logging
logger = logging.getLogger(__name__)


class KNN(Processor[KNNConfig]):
    name: ClassVar[str] = "knn"
    version: ClassVar[int] = 1

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> KNNConfig:
        return KNNConfig(**config)

    def build(self) -> None:
        return

    def _process(self, item: Envelope) -> Envelope | None:
        active = self._require_active(item)
        main = item.tmp[active]

        # load config
        by = item.resolve_cols(self.config.by)
        col = self.config.col

        # edges
        edge_index: list[ArrayI64] = []
        edge_weight: list[ArrayF64] = []

        # iterate over each event dom data
        for row in main.get_column(col).to_list():
            # each row is of shape (N, 3) where N is dom count and 3 for x, y, z coordinates
            # thus this is already in the correct shape for cKDTree
            pos = np.asarray(row, dtype=np.float64)

            # drop any infinite values (unlikely, just in case)
            mask = np.isfinite(pos).all(axis=1)

            # compute edge index/weight
            ei, ew = self.compute_edges(pos[mask])
            edge_index.append(ei)
            edge_weight.append(ew)

        # set up payload
        out = item.resolve_cols(self.config.out)
        if len(out) != 2:
            raise RuntimeError(f"KNN: expected out to have two elements, got {out}")

        # sanity checks
        n = len(main)
        if len(edge_index) != n or len(edge_weight) != n:
            raise RuntimeError(
                f"KNN: length mismatch: payload={n}, edge_index={len(edge_index)}, edge_weight={len(edge_weight)}"
            )

        # payload must include the merge keys
        payload = main.select(by).with_columns(
            pl.Series(out[0], edge_index),
            pl.Series(out[1], edge_weight),
        )

        return item.merge(payload, to=active, on=by, validate="1:1")

    def compute_edges(self, pos: ArrayF64) -> tuple[ArrayI64, ArrayF64]:
        pos = np.ascontiguousarray(pos, dtype=np.float64)

        n = pos.shape[0]
        if n <= 1:
            return np.empty((2, 0), dtype=np.int64), np.empty((0,), dtype=np.float64)

        # get effective k
        k = min(self.config.k, n - 1)

        # build kdtree
        tree = KDTree(pos)

        # query self + k neighbors, then drop self column
        dists_raw, indices_raw = tree.query(pos, k=k + 1)
        dists: ArrayF64 = np.asarray(dists_raw)
        indices: ArrayI = np.asarray(indices_raw)

        if k == 1:
            dists = dists[:, None]
            indices = indices[:, None]

        dists = dists[:, 1:]
        indices = indices[:, 1:]

        src = np.repeat(np.arange(n, dtype=np.int64), k)
        dst = indices.reshape(-1).astype(np.int64, copy=False)

        edge_index = np.empty((2, n * k), dtype=np.int64)
        edge_index[0] = src
        edge_index[1] = dst

        edge_weight = dists.reshape(-1).astype(np.float64, copy=False)
        return edge_index, edge_weight