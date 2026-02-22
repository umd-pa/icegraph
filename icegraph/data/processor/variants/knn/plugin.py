# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import ClassVar, Any

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from icegraph.data.processor import Processor
from icegraph.data.shared.profile import profile_stage
from icegraph.data.types import Envelope
from icegraph.types.common import ArrayF32, ArrayI64

from .config import KNNConfig

__all__ = ["KNN"]


class KNN(Processor[KNNConfig]):
    name: ClassVar[str] = "knn"
    version: ClassVar[int] = 1

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> KNNConfig:
        return KNNConfig(**config)

    def build(self) -> None:
        return

    @profile_stage()
    def _process(self, env: Envelope) -> Envelope | None:
        self._ensure_selected(env)
        main = env.tmp[env.active]

        by = env.resolve_cols(self.config.by)
        cols = env.resolve_cols(self.config.col)

        def _edges_from_row(row: pd.Series):
            pos = np.column_stack(row.to_list()).astype(np.float32, copy=False)
            mask = np.isfinite(pos).all(axis=1)
            return self.compute_edges(pos[mask])

        pairs = main[cols].apply(_edges_from_row, axis=1)
        edge_index, edge_weight = zip(*pairs) if len(pairs) else ([], [])

        # set up payload
        payload = pairs.index.to_frame(index=False)

        out = env.resolve_cols(self.config.out)
        if len(out) != 2:
            raise RuntimeError(f"KNN: expected out to have two elements, got {out}")

        # sanity checks
        n = len(payload)
        if len(edge_index) != n or len(edge_weight) != n:
            raise RuntimeError(
                f"KNN: length mismatch: payload={n}, edge_index={len(edge_index)}, edge_weight={len(edge_weight)}"
            )

        # payload must include the merge keys
        payload = main[by].copy()

        # insert into the payload
        payload[out[0]] = pd.Series(edge_index, index=payload.index, dtype="object")
        payload[out[1]] = pd.Series(edge_weight, index=payload.index, dtype="object")

        return env.merge(payload, to=env.active, on=by, validate="1:1")

    def compute_edges(self, pos: ArrayF32) -> tuple[ArrayI64, ArrayF32]:
        n = pos.shape[0]
        if n <= 1:
            return np.empty((2, 0), dtype=np.int64), np.empty((0,), dtype=np.float32)

        effective_k = min(self.config.k, n - 1)

        nn = NearestNeighbors(n_neighbors=effective_k + 1, algorithm="auto")
        nn.fit(pos)

        distances, indices = nn.kneighbors(pos, return_distance=True)

        # first neighbor is the point itself; drop it
        indices = indices[:, 1:]
        distances = distances[:, 1:]

        # build edge_index in COO format (2, E)
        src = np.repeat(np.arange(n), effective_k)
        dst = indices.reshape(-1)
        edge_index = np.vstack([src, dst]).astype(np.int64, copy=False)
        edge_weight = distances.reshape(-1).astype(np.float32, copy=False)

        return edge_index, edge_weight
