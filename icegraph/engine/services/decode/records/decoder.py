# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from abc import abstractmethod, ABC
from typing import TypeVar, final

import torch
import numpy as np
from torch import Tensor
from jaxtyping import Float, Int

from icegraph.common.plugins import Plugin
from icegraph.common.record import RecordBlock, Column
from icegraph.typing.common import ArrayI

from .types import RecordDecoderContext

__all__ = ["RecordDecoder"]


C = TypeVar("C")


class RecordDecoder(Plugin[C, RecordDecoderContext], ABC):
    """Provides methods for decoding dataset record blocks."""

    @abstractmethod
    def extract(self, block: RecordBlock, key: str) -> Column | None:
        """Look up the raw column stored under a key, or None if absent."""
        ...

    @final
    def extract_features(self, block: RecordBlock, key: str) -> tuple[Float[Tensor, "M F_pre"], ArrayI] | None:
        out = self._extract_features(block, key)

        # not present in the block
        if out is None:
            return None

        features, counts = out

        if features.numel() == 0:
            return None

        # validate: flat node rows plus per-record node counts
        if features.ndim != 2:
            raise ValueError(
                f"Expected 'features' rows with shape [M, F], got shape {tuple(features.shape)}."
            )

        if len(counts) != block.height or int(counts.sum()) != features.shape[0]:
            raise ValueError(
                f"'features' node counts do not cover the rows: "
                f"{len(counts)} records, {int(counts.sum())} counted rows, {features.shape[0]} rows."
            )

        return features, counts

    def _extract_features(self, block: RecordBlock, key: str) -> tuple[Tensor, ArrayI] | None:
        column = self.extract(block, key)

        if column is None:
            return None

        values = column.values
        if values.ndim == 1:
            values = values[:, None]

        return torch.from_numpy(values), column.row_counts

    @final
    def extract_targets(self, block: RecordBlock, key: str) -> Float[Tensor, "B T_pre"] | Int[Tensor, "B T_pre"] | None:
        targets = self._extract_targets(block, key)

        # if not present in the block
        if targets is None or targets.numel() == 0:
            return None

        # validate: one row per record
        if targets.ndim != 2 or targets.shape[0] != block.height:
            raise ValueError(
                f"Expected 'targets' with shape [B={block.height}, T], got shape {tuple(targets.shape)}."
            )

        return targets

    def _extract_targets(self, block: RecordBlock, key: str) -> Tensor | None:
        column = self.extract(block, key)

        if column is None:
            return None

        return torch.from_numpy(self._fold_rows(column, block.height, key))

    @final
    def extract_auxiliary(
            self, block: RecordBlock, key: str
    ) -> Float[Tensor, "B A_pre"] | Int[Tensor, "B A_pre"] | None:
        auxiliary = self._extract_auxiliary(block, key)

        # not present in the block
        if auxiliary is None or auxiliary.numel() == 0:
            return None

        # validate: one row per record
        if auxiliary.ndim != 2 or auxiliary.shape[0] != block.height:
            raise ValueError(
                f"Expected 'auxiliary' with shape [B={block.height}, A], got shape {tuple(auxiliary.shape)}."
            )

        return auxiliary

    def _extract_auxiliary(self, block: RecordBlock, key: str) -> Tensor | None:
        column = self.extract(block, key)

        if column is None:
            return None

        return torch.from_numpy(self._fold_rows(column, block.height, key))

    @final
    def extract_edge_index(self, block: RecordBlock, key: str) -> tuple[Int[Tensor, "2 K"], ArrayI] | None:
        out = self._extract_edge_index(block, key)

        # not present in the block
        if out is None:
            return None

        edge_index, counts = out

        if edge_index.numel() == 0:
            return None

        # validate: unshifted node ids plus per-record edge counts
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(
                f"Expected 'edge_index' with shape [2, K], got shape {tuple(edge_index.shape)}."
            )

        if len(counts) != block.height or int(counts.sum()) != edge_index.shape[1]:
            raise ValueError(
                f"'edge_index' edge counts do not cover the columns: "
                f"{len(counts)} records, {int(counts.sum())} counted edges, {edge_index.shape[1]} edges."
            )

        return edge_index, counts

    def _extract_edge_index(self, block: RecordBlock, key: str) -> tuple[Tensor, ArrayI] | None:
        column = self.extract(block, key)

        if column is None:
            return None

        if column.offsets is None or column.values.ndim != 2 or column.values.shape[1] != 2:
            raise ValueError(
                f"Edge column {key!r} must hold ragged [E, 2] rows, got shape {column.values.shape}."
            )

        values = column.values.astype(np.int64, copy=False)

        # canonical [2, K] orientation; contiguity matters downstream
        return (
            torch.from_numpy(np.ascontiguousarray(values.T)),
            column.row_counts
        )

    @final
    def extract_edge_attr(self, block: RecordBlock, key: str) -> Float[Tensor, "K ATTR"] | None:
        edge_attr = self._extract_edge_attr(block, key)

        # not present in the block
        if edge_attr is None or edge_attr.numel() == 0:
            return None

        # validate: one row per edge
        if edge_attr.ndim != 2:
            raise ValueError(
                f"Expected 'edge_attr' with shape [K, ATTR], got shape {tuple(edge_attr.shape)}."
            )

        return edge_attr

    def _extract_edge_attr(self, block: RecordBlock, key: str) -> Tensor | None:
        column = self.extract(block, key)

        if column is None:
            return None

        values = column.values
        if values.ndim == 1:
            values = values[:, None]

        return torch.from_numpy(values)

    @final
    def extract_simweights(self, block: RecordBlock, key: str) -> Float[Tensor, "B"] | None:
        simweights = self._extract_simweights(block, key)

        # not present in the block
        if simweights is None or simweights.numel() == 0:
            return None

        # validate: one weight per record
        if simweights.ndim != 1 or simweights.shape[0] != block.height:
            raise ValueError(
                f"Expected 'simweights' with shape [B={block.height}], got shape {tuple(simweights.shape)}."
            )

        return simweights

    def _extract_simweights(self, block: RecordBlock, key: str) -> Tensor | None:
        column = self.extract(block, key)

        if column is None:
            return None

        values = column.values.reshape(-1)
        if values.shape[0] != block.height:
            raise ValueError(
                f"Column {key!r} must hold one value per record, "
                f"got {values.shape[0]} for {block.height} records."
            )

        return torch.from_numpy(values)

    @staticmethod
    def _fold_rows(column: Column, height: int, key: str) -> np.ndarray:
        """Reshape a column to one row per record."""
        values = column.values
        counts = column.row_counts

        if values.ndim == 1:
            # rows of scalars: uniform widths fold into a [B, T] table
            width = int(counts[0]) if height else 0
            if not np.all(counts == width):
                raise ValueError(
                    f"Column {key!r} has ragged widths; expected {width} per record."
                )
            return values.reshape(height, width)

        if not np.all(counts == 1):
            raise ValueError(
                f"Column {key!r} must contribute exactly one row per record."
            )
        return values