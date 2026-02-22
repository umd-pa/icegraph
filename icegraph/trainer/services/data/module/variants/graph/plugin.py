# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Iterator, Any, ClassVar
import itertools

from torch import Tensor
from torch_geometric.data import Data
import torch
import numpy as np

from icegraph.types.data import Split, AttributeDomain, ModelInputRole
from icegraph.types.common import ArrayUI8, ArrayF, ArrayG

from ...module import Module

from .config import Config

__all__ = ["GraphModule"]


class GraphModule(Module[Config]):
    """Class for loading IceCube data in PyG Batch format."""
    name: ClassVar[str] = "graph"
    version: ClassVar[int] = 1

    _keys: ArrayUI8 | None

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> Config:
        return Config(**config)

    def build(self) -> None:
        self._keys = None

    def __len__(self) -> int:
        """
        Return the number of events in the subset.

        Returns:
            int: Number of events.
        """
        return len(self.keys)

    def _load_keys(self) -> ArrayUI8:
        # load the splitmap from dataset attrs
        splitmaps: Iterator[list[int]] = (
            attr[AttributeDomain.LOCAL]["splitmap"] for attr in self._store.attrs
        )

        # split keys are small integers (0,1,2), so uint8 is safe and more memory friendly
        splitmap = np.fromiter(itertools.chain.from_iterable(splitmaps), dtype=np.uint8)

        # build the mask and return
        return np.where(splitmap == self._split.to_int())[0].astype(np.uint8, copy=False)

    @property
    def keys(self) -> ArrayUI8:
        """
        Load the filtered-by-subset key list.

        Returns:
            NDArray: An array of keys.
        """
        if self._keys is None:
            self._keys = self._load_keys()
        return self._keys

    def _get_by_role(self, record: dict[str, Any], role: ModelInputRole, index: int, *, filter_to: list[str]) -> Tensor:
        # get the array
        array: ArrayF = record.get(role.value)

        # ensure it exists
        if array is None:
            raise KeyError(f"Record at index {index} missing '{role.value}' column.")

        if filter_to:
            columns = self._store.global_attrs.columns(role)
            mask = [c in filter_to for c in columns]

            # filter the array
            array = array[:, mask]

        # normalize to tensor
        tensor = torch.tensor(array)

        # ensure features/labels has the correct shape [1, F/L]
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)

        return tensor

    @staticmethod
    def _get_by_key(record: dict[str, Any], key: str, index: int, *, dtype: torch.dtype | None = None) -> Tensor:
        # get the array
        array: ArrayG = record.get(key)

        # ensure it exists
        if array is None:
            raise KeyError(f"Record at index {index} missing '{key}' column.")

        # normalize to tensor and return
        return torch.tensor(array, dtype=dtype)

    def get(self, index: int) -> Data:
        """
        Read and decode a single event from the dataset.

        Args:
            index (int): Global index of the event to retrieve.
        """
        # fetch the record from the shard store
        record = self._store[index]

        # load features and labels
        features    = self._get_by_role(record, ModelInputRole.FEATURES, index, filter_to=self.config.features)
        labels      = self._get_by_role(record, ModelInputRole.LABELS, index, filter_to=self.config.labels)

        # load edge indices and weights
        edge_index  = self._get_by_key(record, "edge_index", index, dtype=torch.long)
        edge_attr   = self._get_by_key(record, "edge_weight", index, dtype=torch.float32)

        # ensure correct dim
        if edge_index.ndim != 2:
            raise ValueError(
                f"Value 'edge_index' must be 2D (shape [2, E]), got {edge_index.ndim}D at index {index}."
            )

        # ensure shape [2, E]
        if edge_index.shape[0] != 2:
            raise ValueError(
                f"Value 'edge_index' must have shape [2, E], got {tuple(edge_index.shape)} at index {index}."
            )

        # ensure correct dim
        if edge_attr.ndim != 1:
            raise ValueError(
                f"Value 'edge_attr' must be 1D (shape [E]), got {edge_attr.ndim}D at index {index}."
            )

        # ensure edge attrs and index match
        if edge_index.shape[1] != edge_attr.shape[0]:
            raise ValueError(
                f"edge count mismatch: edge_index has shape [2, E] = {edge_index.shape} edges "
                f"but edge_attr has shape [E] = {edge_attr.shape} at index {index}"
            )

        # populate payload
        payload: dict[str, Tensor] = {
            "x": features, "y": labels, "edge_index": edge_index, "edge_attr": edge_attr
        }

        # build the torch geometric Data object
        data = Data(**payload)

        # if in eval, append auxiliary label data
        if self._split in Split.eval():
            data.aux = self._get_by_role(record, ModelInputRole.FEATURES, index, filter_to=self.config.aux)

        return data
