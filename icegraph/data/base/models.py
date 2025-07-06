# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import Union, Optional
from pathlib import Path
from abc import ABC
import functools
import lmdb
import struct
import msgpack

from torch_geometric.data import Dataset, DataLoader
import torch_geometric as pyg
import torch
import numpy as np

from icegraph.data.preprocess import generate_vector_mapping
from icegraph.config import IGConfig

__all__ = ["IGData"]


class IGData(Dataset, ABC):
    """
    The base dataset class for loading and managing IceCube data stored in LMDB format.

    This class handles the truth table, feature loading, and optional selection filtering
    for training, validation, or test subsets. Subclasses must set the class attribute `subset`
    to one of: "train", "validation", or "test".
    """

    subset: Optional[str] = None

    dataloader = property(
        lambda self: functools.partial(DataLoader, self),
        doc="A convenience property that returns a partially-applied torch geometric DataLoader constructor for this dataset."
    )

    def __init__(self, input_file: Union[str, Path]) -> None:
        """
        Initialize an IGData object from an LMDB file.

        Args:
            input_file (Union[str, Path]): Path to the input file (LMDB).
        """
        super().__init__()
        self.input_file = Path(input_file)
        self._config: IGConfig = IGConfig.get()

        self.env = lmdb.open(
            str(self.input_file),
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False
        )
        self.txn = self.env.begin(write=False)
        self.keys = list(range(self._get_length()))

        # load target labels once on instantiation
        self.target_labels = self._config.user_config.data.target_labels

        self.features_columns = list(generate_vector_mapping().values())

    def __init_subclass__(cls, **kwargs) -> None:
        """
        Validate that subclasses of IGData define a proper `subset` attribute.

        Ensures that any subclass sets the class-level `subset` to one of
        the allowed split names ("train", "validation", or "test"), and that
        it consists only of alphabetic characters.

        Raises:
            TypeError: If `subset` is not defined on the subclass.
            ValueError: If `subset` is not an alphabetic string.
        """
        super().__init_subclass__(**kwargs)
        if cls.subset is None:
            raise TypeError(f"{cls.__name__}.subset must be set to 'train'|'validation'|'test'")
        if not isinstance(cls.subset, str) or not cls.subset.isalpha():
            raise ValueError(f"`subset` on {cls.__name__!r} must be alpha string, got {cls.subset!r}")

    def __getitem__(self, idx: int) -> pyg.data.Data:  # writing as pyg.data.Data for clarity
        """
        Retrieve a single sample by index.

        Args:
            idx (int): Index of the event.

        Returns:
            pyg.data.Data: Torch-Geometric Data object containing data for the selected event.
        """
        features, labels, edge_index, edge_weight = self.get(idx)
        return pyg.data.Data(x=features, y=labels, edge_index=edge_index, edge_attr=edge_weight)

    def __len__(self) -> int:
        """
        Return the number of events in the subset.

        Returns:
            int: Number of events.
        """
        return len(self.keys)

    def _get_length(self) -> int:
        """
        Return the number of events in the subset.

        Returns:
            int: Number of events.
        """
        with self.env.begin() as txn:
            stat = txn.stat()
            return stat['entries']

    def get(self, idx: int) -> tuple[torch.Tensor, ...]:
        """
        Read and decode a single event from the LMDB dataset.

        Args:
            idx (int): Index of the event to retrieve.

        Returns:
            tuple[torch.Tensor, ...]: Tuple containing:
                - features (torch.Tensor): Node features of shape [num_nodes, num_features].
                - labels (torch.Tensor): Target labels of shape [num_labels].
                - edge_index (torch.Tensor): Edge indices of shape [2, num_edges].
                - edge_weight (torch.Tensor): Edge weights of shape [num_edges].

        Raises:
            IndexError: If the sample key is not found in the LMDB.
        """
        # convert to tensors
        key = struct.pack(">Q", self.keys[idx])
        value = self.txn.get(key)

        if value is None:
            raise IndexError(f"Key {self.keys[idx]} not found in {self.input_file!s}.")

        data = msgpack.unpackb(value, raw=False)

        features_dict = data["features"]
        key_order = list(features_dict[0].keys())

        features = np.array([[d[k] for k in key_order] for d in features_dict], dtype=np.float32)
        labels = np.array([data[label] for label in self.target_labels], dtype=np.float32)
        edge_index = np.array(data["edge_index"], dtype=np.float32)
        edge_weight = np.array(data["edge_weight"], dtype=np.float32)

        return tuple(map(torch.from_numpy, (features, labels, edge_index, edge_weight)))
