# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import Union, Optional, Literal, Sequence, ClassVar, Dict, Any, List, Type
from pathlib import Path
from abc import ABC
import functools
import os

from torch.utils.data import Dataset
import torch_geometric as pyg
from torch_geometric.data import Data as PyGData
import torch
import numpy as np
import pandas as pd

from icegraph.config import IGConfig
from icegraph.data.base.exceptions import EmptyDatasetError, DataError, MissingFieldError
from icegraph.data.readers import LMDBDatasetShardReader, LMDBReader
from icegraph.utils import stable_hash_cbor

__all__ = ["IGData"]


class IGData(Dataset, ABC):
    """
    The base dataset class for loading and managing IceCube data stored in LMDB format.

    This class handles the truth table, feature loading, and optional selection filtering
    for training, validation, or test subsets. Subclasses must set the class attribute `subset`
    to one of: "train", "validation", or "test".
    """

    # subset defined in each subclass
    subset:             ClassVar[Optional[Literal["train","validation","test"]]]            = None

    # shared class vars
    _reader:            ClassVar[Optional[Type[LMDBDatasetShardReader]]]                          = None
    _source:            ClassVar[Optional[Union[str, Path, Sequence[Union[str, Path]]]]]    = None

    # reader process id for forks
    _reader_pid:        ClassVar[Optional[int]]                                             = None

    # dataset attributes
    attrs:              ClassVar[Optional[Dict[int, Any]]]                                  = None

    dataloader = property(
        lambda self: functools.partial(pyg.loader.DataLoader, self),
        doc="A convenience property that returns a partially-applied torch geometric DataLoader constructor for this dataset."
    )

    def __init__(self) -> None:
        """
        Initialize an IGData object from an LMDB file.

        Raises:
            EmptyDatasetError: If the loaded dataset is empty.
        """
        # verify preloading attributes have been set
        cls = type(self)
        if cls._source is None:
            raise DataError("Please run IGData.configure() before instantiating subclasses.")

        super().__init__()

        # build the reader
        cls._build_reader()

        # grab global config
        self._config: IGConfig = IGConfig.get()

        # get the key list
        self._keys: Optional[List[int]] = None

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

    def __getitem__(self, idx: int) -> PyGData:
        """
        Retrieve a single sample by index.

        Args:
            idx (int): Index of the event.

        Returns:
            pyg.data.Data: Torch-Geometric Data object containing data for the selected event.
        """
        x, y, edge_index, edge_weight, include_labels = self.get(idx)
        data = PyGData(x=x, y=y, edge_index=edge_index, edge_attr=edge_weight)

        if include_labels is not None:
            data.include_labels = include_labels

        return data

    def __len__(self) -> int:
        """
        Return the number of events in the subset.

        Returns:
            int: Number of events.
        """
        return len(self.keys)

    @classmethod
    def _build_reader(cls) -> None:
        """
        Build the shared shard reader to be used by all subclasses.
        """
        pid = os.getpid()
        if cls._reader is None or cls._reader_pid != pid:
            cls._reader = LMDBDatasetShardReader

    def _load_keys(self) -> List[int]:
        # load the split key for the subset
        split_key = self._config.internal_config.split_int_assignments[self.subset]

        # load the splitmap from dataset attrs
        splitmap = self._load_splitmap()

        # build the mask and return
        mask = (splitmap == split_key)
        return np.where(mask)[0]

    def _load_splitmap(self) -> List[int]:
        splitmaps: List[List[int]] = []

        with self._reader() as reader:
            attrs = reader.attrs()
            for i in range(len(attrs)):
                splitmaps.append(attrs[i]["allocation"]["splitmap"])

        flattened_splitmap = np.array([x for _map in splitmaps for x in _map])
        return flattened_splitmap

    @property
    def keys(self) -> List[int]:
        if self._keys is None:
            self._keys = self._load_keys()
        return self._keys

    @classmethod
    def configure(cls, source: Union[str, Path, Sequence[Union[str, Path]]]) -> None:
        """
        Set the dataset configurations.

        Args:
            source (Union[str, Path, Sequence[Union[str, Path]]]): Path to the input file(s) (LMDB) or a directory.
        """
        if cls._source is None:
            cls._source = source

        # configure the reader class
        LMDBDatasetShardReader.configure(
            source=cls._source,
            max_open_envs=4,
            clean=True
        )

        # load metadata
        with LMDBDatasetShardReader() as reader:
            cls.attrs = reader.attrs()

        # verify config hash
        config = cls.attrs[0]["global"]["config"]
        config_hash = cls.attrs[0]["global"]["config_hash"]

        if config_hash != stable_hash_cbor(config):
            raise RuntimeError("Source config hash does not match expected hash. One or more files may be corrupted.")

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
        elif y.ndim == 2:
            return len(y[0])
        else:
            raise DataError(
                f"Expected label tensor to be 0D, 1D, or 2D, got {y.ndim}D (shape={tuple(y.shape)})"
            )

    @property
    def num_node_features(self):
        """
       Returns the dimensionality of the input feature list for each graph sample.

       Returns:
           int: The number of input features.
       """
        return self[0].x.size(-1)

    def get(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Read and decode a single event from the LMDB dataset.

        Args:
            idx (int): Index of the event to retrieve.

        Returns:
            tuple[torch.Tensor, ...]: Tuple containing:
                - features (torch.Tensor): Node features of shape [num_nodes, num_features].
                - labels (torch.Tensor): Target labels of shape [1, num_labels].
                - edge_index (torch.Tensor): Edge indices of shape [2, num_edges].
                - edge_weight (torch.Tensor): Edge weights of shape [num_edges].
                - included_labels (Optional[torch.Tensor]): Included labels of shape [1, num_included_labels].

        Raises:
            MissingFieldError, DataError
        """
        cls = type(self)

        if not isinstance(idx, int):
            raise DataError(f"Index must be int, got {type(idx).__name__!r}")

        # fetch the record from the reader
        try:
            with cls._reader() as reader:
                data, *_ = reader[idx]
        except (IndexError, KeyError) as e:
            raise DataError(f"Failed to retrieve record at index {idx}: {e}")

        # --- features ---
        try:
            features = torch.tensor(data["features"], dtype=torch.float32)
        except KeyError:
            raise MissingFieldError(f"Record at index {idx} missing 'features' field")
        if features.ndim == 1:
            features = features.unsqueeze(0)  # [F] -> [1, F]

        # --- labels ---
        labels_vals = []
        for name in cls.attrs[0]["global"]["target_labels"]:
            if name not in data:
                raise MissingFieldError(f"Label '{name}' not found in record at index {idx}")
            labels_vals.append(data[name])
        labels = torch.tensor(labels_vals, dtype=torch.float32).unsqueeze(0)  # [1, L]

        # --- included labels (val/test only) ---
        included_labels: Optional[torch.Tensor] = None
        if self.subset in ["validation", "test"]:
            inc_vals = []
            for name in cls.attrs[0]["global"]["include_labels"]:
                if name in cls.attrs[0]["global"]["target_labels"]:
                    continue
                if name not in data:
                    raise MissingFieldError(f"Included label '{name}' not found in record at index {idx}")
                inc_vals.append(data[name])
            if inc_vals:
                included_labels = torch.tensor(inc_vals, dtype=torch.float32).unsqueeze(0)  # [1, N_inc]

        # --- edges ---
        try:
            edge_index = torch.tensor(data["edge_index"], dtype=torch.long)
        except KeyError:
            raise MissingFieldError(f"Record at index {idx} missing 'edge_index' field")
        try:
            edge_weight = torch.tensor(data["edge_weight"], dtype=torch.float32)
        except KeyError:
            raise MissingFieldError(f"Record at index {idx} missing 'edge_weight' field")

        # normalize shapes / sanity checks
        if edge_index.ndim != 2:
            raise DataError(f"'edge_index' must be 2-D, got {tuple(edge_index.shape)} at index {idx}")
        if edge_index.shape[0] == 2:
            pass  # [2, E]
        elif edge_index.shape[1] == 2:
            edge_index = edge_index.T.contiguous()  # [E,2] -> [2,E]
        else:
            raise DataError(f"'edge_index' must be [2, E] or [E, 2], got {tuple(edge_index.shape)} at index {idx}")

        if edge_weight.ndim == 2 and edge_weight.shape[1] == 1:
            edge_weight = edge_weight.reshape(-1)
        if edge_weight.ndim != 1:
            raise DataError(f"'edge_weight' must be 1-D (or [E,1]), got {tuple(edge_weight.shape)} at index {idx}")

        if edge_index.shape[1] != edge_weight.shape[0]:
            raise DataError(
                f"edge count mismatch: edge_index has {edge_index.shape[1]} edges "
                f"but edge_weight has {edge_weight.shape[0]} at index {idx}"
            )

        return features, labels, edge_index, edge_weight, included_labels
