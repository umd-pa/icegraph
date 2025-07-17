# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import Union, Optional
from pathlib import Path
from abc import ABC
import functools
import lmdb
import struct
import msgpack

from torch_geometric.data import Dataset
import torch_geometric as pyg
import torch
import numpy as np

from icegraph.data.processor import generate_vector_mapping
from icegraph.config import IGConfig
from icegraph.pathutils import PathValidator
from icegraph.data.base.exceptions import EmptyDatasetError, DataError, MissingFieldError

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
        lambda self: functools.partial(pyg.loader.DataLoader, self),
        doc="A convenience property that returns a partially-applied torch geometric DataLoader constructor for this dataset."
    )

    def __init__(self, infile: Union[str, Path]) -> None:
        """
        Initialize an IGData object from an LMDB file.

        Args:
            infile (Union[str, Path]): Path to the input file (LMDB).

        Raises:
            EmptyDatasetError: If the loaded dataset is empty.
        """
        super().__init__()
        self.infile = Path(infile)
        PathValidator.is_valid_file(self.infile)

        # grab global config
        self._config: IGConfig = IGConfig.get()

        self.env = lmdb.open(
            str(self.infile),
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False
        )

        # build a flat list of keys
        n_entries = self._txn().stat()["entries"]
        if n_entries == 0:
            raise EmptyDatasetError(f"Dataset at {self.infile!s} is empty.")
        self.keys = list(range(n_entries))

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
            raise DataError(f"{cls.__name__}.subset must be set to 'train'|'validation'|'test'")
        if not isinstance(cls.subset, str) or not cls.subset.isalpha():
            raise DataError(f"`subset` on {cls.__name__!r} must be alpha string, got {cls.subset!r}")

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

    def __del__(self):
        try:
            self.env.close()
        except Exception:
            pass

    def _txn(self):
        """
        Thread-safe txn.
        """
        if not hasattr(self, "_thread_txn"):
            self._thread_txn = self.env.begin(write=False)
        return self._thread_txn

    @property
    def num_output_features(self) -> int:
        """
        Returns the dimensionality of the target (label) for each graph sample.

        Returns:
            int: The number of output features (e.g., 1 for scalar regression,
                 or the number of classes/targets for vector labels).
        """
        y = self[0].y
        if y.ndim == 0:
            return 1
        elif y.ndim == 1:
            return y.shape[0]
        else:
            raise DataError(
                f"Expected label tensor to be 0D or 1D, got {y.ndim}D (shape={tuple(y.shape)})"
            )

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
            MissingFieldError, DataError
        """
        # verify passed index
        if not isinstance(idx, int):
            raise DataError(f"Index must be int, got {type(idx).__name__!r}")
        if idx < 0 or idx >= len(self.keys):
            raise DataError(f"Index {idx} out of bounds for dataset of size {len(self.keys)}")
        key_id = self.keys[idx]

        # pack key and fetch
        packed = struct.pack(">Q", key_id)
        value = self._txn().get(packed)

        # quick check value
        if value is None:
            raise DataError(f"Key {key_id} not found in LMDB '{self.infile}'")

        # unpack msgpack
        try:
            data = msgpack.unpackb(value, raw=False)
        except Exception as e:
            raise DataError(f"Corrupt record {key_id}: {e}")

        # get features list
        try:
            features_list = data["features"]
        except KeyError:
            raise MissingFieldError(f"Record {key_id!r} missing 'features' field")

        key_order = list(features_list[0].keys())

        # build feature array
        features_np = np.array(
            [[feat[k] for k in key_order] for feat in features_list],
            dtype=np.float32
        )

        # build labels array
        labels_vals = []
        for name in self.target_labels:
            if name not in data:
                raise MissingFieldError(f"Label '{name}' not found in record for key {key_id}")
            labels_vals.append(data[name])
        labels_np = np.array(labels_vals, dtype=np.float32)

        # build edge index/weight
        try:
            ei_list = data["edge_index"]
        except KeyError:
            raise MissingFieldError(f"Record {key_id!r} missing 'edge_index' field")
        try:
            ew_list = data["edge_weight"]
        except KeyError:
            raise MissingFieldError(f"Record {key_id!r} missing 'edge_weight' field")

        ei_np = np.array(ei_list, dtype=np.int64)
        ew_np = np.array(ew_list, dtype=np.float32)

        # assert dims are what we expect
        assert ei_np.ndim == 2 and ei_np.shape[0] == 2, (
            f"edge_index for key {key_id!r} must be [2, E], got {ei_np.shape}"
        )

        # convert everything to tensors
        features = torch.from_numpy(features_np)
        labels = torch.from_numpy(labels_np)
        edge_index = torch.from_numpy(ei_np).long()
        edge_weight = torch.from_numpy(ew_np)

        return features, labels, edge_index, edge_weight
