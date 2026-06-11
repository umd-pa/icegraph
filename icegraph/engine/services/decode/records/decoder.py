# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from abc import abstractmethod, ABC
from typing import TypeVar, final

import torch
from torch import Tensor
from jaxtyping import Float, Int

from icegraph.common.plugins import Plugin
from icegraph.common.record import Record

from .types import RecordDecoderContext

__all__ = ["RecordDecoder"]


C = TypeVar("C")


class RecordDecoder(Plugin[C, RecordDecoderContext], ABC):
    """Provides methods for decoding dataset records."""

    @abstractmethod
    def extract(self, record: Record, key: str) -> Tensor:
        ...

    @final
    def extract_features(self, record: Record, key: str) -> Float[Tensor, "N F_pre"]:
        features = self._extract_features(record, key)

        # normalize shape
        if features.ndim == 1:
            # assume [F] -> [1, F]
            return features.unsqueeze(0)

        if features.ndim == 2:
            # already [N, F]
            return features

        raise ValueError(
            f"Expected 'features' tensor with shape [F] or [N, F], "
            f"got shape {tuple(features.shape)}."
        )

    def _extract_features(self, record: Record, key: str) -> Float[Tensor, "F_pre"] | Float[Tensor, "N F_pre"]:
        return self.extract(record, key)

    @final
    def extract_targets(self, record: Record, key: str) -> Float[Tensor, "1 T_pre"] | Int[Tensor, "1 T_pre"]:
        targets = self._extract_targets(record, key)

        # normalize shape
        if targets.ndim == 1:
            # assume [T] -> [1, T]
            return targets.unsqueeze(0)

        if targets.ndim == 2 and targets.shape[0] == 1:
            # already [1, T]
            return targets

        raise ValueError(
            f"Expected 'targets' tensor with shape [T] or [1, T], "
            f"got shape {tuple(targets.shape)}."
        )

    def _extract_targets(
            self, record: Record, key: str
    ) -> Float[Tensor, "W_pre"] | Int[Tensor, "W_pre"] | Float[Tensor, "1 W_pre"] | Int[Tensor, "1 W_pre"]:
        return self.extract(record, key)

    @final
    def extract_auxiliary(self, record: Record, key: str) -> Float[Tensor, "1 A_pre"] | Int[Tensor, "1 A_pre"]:
        auxiliary = self._extract_auxiliary(record, key)

        # auxiliary is allowed to be empty
        if auxiliary.numel() == 0:
            return torch.empty((1, 0), dtype=torch.int64)

        # normalize shape
        if auxiliary.ndim == 1:
            # assume [A] -> [1, A]
            return auxiliary.unsqueeze(0)

        if auxiliary.ndim == 2 and auxiliary.shape[0] == 1:
            # already [1, A]
            return auxiliary

        raise ValueError(
            f"Expected 'auxiliary' tensor with shape [A] or [1, A], "
            f"got shape {tuple(auxiliary.shape)}."
        )

    def _extract_auxiliary(
            self, record: Record, key: str
    ) -> Float[Tensor, "A_pre"] | Int[Tensor, "A_pre"] | Float[Tensor, "1 A_pre"] | Int[Tensor, "1 A_pre"] | Int[Tensor, "0"]:
        return self.extract(record, key)

    @final
    def extract_edge_index(self, record: Record, key: str) -> Int[Tensor, "2 E"]:
        edge_index = self._extract_edge_index(record, key)

        # edge_index is allowed to be empty
        if edge_index.numel() == 0:
            return torch.empty((2, 0), dtype=torch.int64)

        # normalize / validate shape
        if edge_index.ndim == 2 and edge_index.shape[0] == 2:
            # already [2, E]
            return edge_index

        raise ValueError(
            f"Expected 'edge_index' tensor with shape [2, E], "
            f"got shape {tuple(edge_index.shape)}."
        )

    def _extract_edge_index(self, record: Record, key: str) -> Int[Tensor, "2 E"]:
        return self.extract(record, key)

    @final
    def extract_edge_attr(self, record: Record, key: str) -> Float[Tensor, "E ATTR"]:
        edge_attr = self._extract_edge_attr(record, key)

        # edge_attr is allowed to be empty
        if edge_attr.numel() == 0:
            return torch.empty((0, 0), dtype=torch.float32)

        # normalize / validate shape
        if edge_attr.ndim == 2:
            # already [E, ATTR]
            return edge_attr

        if edge_attr.ndim == 1:
            # [E] -> [E, 1]
            return edge_attr.unsqueeze(-1)

        raise ValueError(
            f"Expected 'edge_attr' tensor with shape [E, ATTR] or [E], "
            f"got shape {tuple(edge_attr.shape)}."
        )

    def _extract_edge_attr(self, record: Record, key: str) -> Float[Tensor, "E ATTR"]:
        return self.extract(record, key)

    @final
    def extract_simweights(self, record: Record, key: str) -> Float[Tensor, "1"] | Float[Tensor, "0"]:
        simweights = self._extract_simweights(record, key)

        # simweights is allowed to be empty
        if simweights.numel() == 0:
            return torch.empty((0,), dtype=torch.float32)

        # normalize / validate shape
        if simweights.ndim == 0:
            # scalar -> [1]
            return simweights.unsqueeze(0)

        if simweights.ndim == 1 and simweights.shape[0] == 1:
            # already [1]
            return simweights

        raise ValueError(
            f"Expected 'simweights' tensor with shape [1], "
            f"got shape {tuple(simweights.shape)}."
        )

    def _extract_simweights(
            self, record: Record, key: str
    ) -> Float[Tensor, ""] | Float[Tensor, "1"] | Float[Tensor, "0"]:
        return self.extract(record, key)