# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import Iterator
import itertools
from collections.abc import Sized

from torch import Tensor
from torch.utils.data import Dataset
from torch_geometric.data import Data
import torch
import numpy as np

from icegraph.config import IGConfig
from icegraph.types.data import Split, AttributeDomain
from icegraph.types.common import ArrayI, ArrayG

from .readers import ShardStore

__all__ = ["DatasetModule"]

# module logger
import logging
logger = logging.getLogger(__name__)


class DatasetModule(Dataset[Data], Sized):
    """The base dataset class for loading and managing IceCube data."""

    def __init__(self, split: Split, store: ShardStore) -> None:
        super().__init__()

        # cache split assignment and store reference
        self.split = split
        self.store = store

        # grab global config
        self._config = IGConfig.get()

        # cache for the key list
        self._keys: ArrayI | None = None

        # keys cache
        self.auxiliary_labels:  list[str] = self._config.user_config.training.auxiliary_labels
        self.target_labels:     list[str] = self._config.user_config.training.target_labels

        logger.debug(
            "initialized %s (split=%s) with target_labels=%s, auxiliary_labels=%s",
            type(self).__name__, self.split.name, self.target_labels, self.auxiliary_labels
        )

    def __getitem__(self, i: int) -> Data:
        """
        Retrieve a single sample by index.

        Args:
            i (int): Index of the event.
        """
        # delegate to get
        return self.get(i)

    def __len__(self) -> int:
        """
        Return the number of events in the subset.

        Returns:
            int: Number of events.
        """
        return len(self.keys)

    def _load_keys(self) -> ArrayI:
        # load the splitmap from dataset attrs
        splitmaps: Iterator[list[int]] = (
            attr[AttributeDomain.LOCAL]["splitmap"] for attr in self.store.attrs
        )

        # split keys are small integers (0,1,2), so uint8 is safe and more memory friendly
        splitmap = np.fromiter(itertools.chain.from_iterable(splitmaps), dtype=np.uint8)

        # build the mask and return
        return np.where(splitmap == self.split.to_int())[0]

    @property
    def keys(self) -> ArrayI:
        """
        Load the filtered-by-subset key list.

        Returns:
            NDArray: An array of keys.
        """
        if self._keys is None:
            self._keys = self._load_keys()
        return self._keys

    @staticmethod
    def _get_features(record: dict[str, ArrayG], i: int) -> Tensor:
        # get features
        features = record.get("features")

        # ensure features exists
        if features is None:
            raise KeyError(f"Record at index {i} missing 'features' column.")

        # normalize to tensor
        features_t = torch.tensor(features, dtype=torch.float32)

        # ensure features has the correct shape [1, F]
        if features_t.ndim == 1:
            features_t = features_t.unsqueeze(0)

        return features_t

    @staticmethod
    def _get_labels(record: dict[str, ArrayG], target_labels: list[str], i: int) -> Tensor:
        # get labels
        labels: list[ArrayG] = []

        for label in target_labels:
            # load the column data
            column = record.get(label)

            # ensure label column exists
            if column is None:
                raise KeyError(f"Record at index {i} missing '{label}' column.")

            # append to accumulator
            labels.append(column)

        # normalize to tensor
        labels_t = torch.tensor(labels, dtype=torch.float32)

        # ensure labels has the correct shape [1, L]
        if labels_t.ndim == 1:
            labels_t = labels_t.unsqueeze(0)

        return labels_t

    @staticmethod
    def _get_edge_index(record: dict[str, ArrayG], i: int) -> Tensor:
        # get edge_index
        edge_index = record.get("edge_index")

        # ensure edge_index column exists
        if edge_index is None:
            raise KeyError(f"Record at index {i} missing 'edge_index' column.")

        # normalize to tensor
        edge_index_t = torch.tensor(edge_index, dtype=torch.long)

        # ensure correct dim
        if edge_index_t.ndim != 2:
            raise ValueError(
                f"Value 'edge_index' must be 2D (shape [2, E]), got {edge_index_t.ndim}D at index {i}."
            )

        # ensure shape [2, E]
        if edge_index_t.shape[0] != 2:
            raise ValueError(
                f"Value 'edge_index' must have shape [2, E], got {tuple(edge_index_t.shape)} at index {i}."
            )

        return edge_index_t

    @staticmethod
    def _get_edge_attr(record: dict[str, ArrayG], i: int) -> Tensor:
        # get edge_attr (stored as edge_weight in lmdb)
        edge_attr = record.get("edge_weight")

        # ensure edge_index column exists
        if edge_attr is None:
            raise KeyError(f"Record at index {i} missing 'edge_weight' column.")

        # normalize to tensor
        edge_attr_t = torch.tensor(edge_attr, dtype=torch.float32)

        # ensure correct dim
        if edge_attr_t.ndim != 1:
            raise ValueError(
                f"Value 'edge_weight' must be 1D (shape [E]), got {edge_attr_t.ndim}D at index {i}."
            )

        return edge_attr_t

    def get(self, i: int) -> Data:
        """
        Read and decode a single event from the LMDB dataset.

        Args:
            i (int): Global index of the event to retrieve.
        """
        if not isinstance(i, int):
            raise TypeError(f"Index must be int, got {type(i).__name__}")

        # fetch the record from the shard store
        index = int(self.keys[i])  # convert to split key
        record = self.store[index]

        # populate payload
        payload: dict[str, Tensor] = {
            "x": self._get_features(record, index),
            "y": self._get_labels(record, self.target_labels, index),
            "edge_index": self._get_edge_index(record, index),
            "edge_attr": self._get_edge_attr(record, index)
        }

        # ensure edge attrs and index match
        if payload["edge_index"].shape[1] != payload["edge_attr"].shape[0]:
            raise ValueError(
                f"edge count mismatch: edge_index has shape [2, E] = {payload['edge_index'].shape} edges "
                f"but edge_attr has shape [E] = {payload['edge_attr'].shape} at index {index}"
            )

        # build the torch geometric Data object
        data = Data(**payload)

        # if not in training split (on eval), append auxiliary label data
        if self.split != Split.TRAIN:
            data.aux = self._get_labels(record, self.auxiliary_labels, index)

        return data
