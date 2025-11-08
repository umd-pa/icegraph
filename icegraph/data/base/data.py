# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

import os
from typing import Union, Optional, Literal, Sequence, Dict, Any, List, Tuple, TypeVar, Callable, cast
from pathlib import Path
import functools
import itertools

from torch.utils.data import Dataset
import torch_geometric as pyg
from torch_geometric.data import Data as PyGData
import torch
import numpy as np
from numpy.typing import NDArray

from icegraph.config import IGConfig
from icegraph.data.base.exceptions import DataError, MissingFieldError
from icegraph.data.readers import LMDBDatasetShardReader

__all__ = ["DataModule"]


F = TypeVar("F", bound=Callable[..., Any])


class DataModule(Dataset):
    """
    The base dataset class for loading and managing IceCube data stored in LMDB format.

    This class handles the truth table, feature loading, and optional selection filtering
    for training, validation, or test subsets.
    """

    dataloader = property(
        lambda self: functools.partial(pyg.loader.DataLoader, self),
        doc="A convenience property that returns a partially-applied torch geometric DataLoader constructor for this dataset."
    )

    def __init__(
            self,
            source: Union[str, Path, Sequence[Union[str, Path]]],
            subset: Literal['train', 'validation', 'test']
    ) -> None:
        """
        Initialize a DataModule object from an LMDB source.
        """
        super().__init__()

        # data source and subset assignment
        self._source:       Union[str, Path, Sequence[Union[str, Path]]]    = source
        self.subset:        Literal['train', 'validation', 'test']          = subset

        # reader attrs
        self._proc_pid:     Optional[int]           = None
        self._reader:       LMDBDatasetShardReader

        # grab global config
        self._config: IGConfig = IGConfig.get()

        # get the key list
        self._keys: Optional[np.ndarray] = None

        # keys cache
        self.include_labels:    Optional[List[str]] = None
        self.target_labels:     Optional[List[str]] = None

    def __getitem__(self, idx: int) -> PyGData:
        """
        Retrieve a single sample by index.

        Args:
            idx (int): Index of the event.

        Returns:
            pyg.data.Data: Torch-Geometric Data object containing data for the selected event.
        """
        data_dict, attr_dict = self.get(cast(int, self.keys[idx]))
        data = PyGData(**data_dict)

        for attr, value in attr_dict.items():
            setattr(data, attr, value)

        return data

    def __len__(self) -> int:
        """
        Return the number of events in the subset.

        Returns:
            int: Number of events.
        """
        return len(self.keys)

    def __del__(self):
        # best-effort cleanup
        try:
            self.close()
        except Exception:
            pass

    def _load_keys(self) -> NDArray:
        # load the split key for the subset
        split_key = self._config.internal_config.split_int_assignments[self.subset]

        # load the splitmap from dataset attrs
        splitmap = self._load_splitmap()

        # build the mask and return
        return np.where((splitmap == split_key))[0]

    def _load_splitmap(self) -> NDArray[np.uint8]:
        """
        Load the split mapping for the dataset.

        Returns:
            NDArray[np.uint8]: Array of keys as unsigned int8.
        """
        # split keys are small integers (0,1,2), so uint8 is safe and more compact
        splitmap = np.fromiter(
            itertools.chain.from_iterable(attr["allocation"]["splitmap"] for attr in self.attrs.values()),
            dtype=np.uint8
        )

        return splitmap

    def _ensure_reader(self):
        """Load the dataset shard reader, ensures a unique one for each process."""
        pid = os.getpid()
        if getattr(self, "_reader", None) is not None and pid == self._proc_pid:
            return self._reader

        # create per-process reader
        self._proc_pid = pid
        self._reader = LMDBDatasetShardReader(self._source)

        return self._reader

    def close(self) -> None:
        """Close the instance."""
        self._reader.close()

    @property
    def attrs(self) -> Dict[bytes, Dict[str, Dict[str, Any]]]:
        self._ensure_reader()
        return self._reader.attrs()

    @property
    def keys(self) -> NDArray:
        """
        Load the filtered-by-subset key list.

        Returns:
            NDArray: An array of keys.
        """
        if self._keys is None:
            self._keys = self._load_keys()
        return self._keys

    @property
    def num_target_labels(self) -> int:
        """
        Returns the dimensionality of the target (label) for each graph sample.

        Returns:
            int: The number of output labels.
        """
        y = self[0].y
        if y.ndim == 0:
            return 1
        elif y.ndim == 1:
            return y.shape[0]
        elif y.ndim == 2:
            return y.shape[1]
        else:
            raise DataError(
                f"Expected label tensor to be 0D, 1D, or 2D, got {y.ndim}D (shape={tuple(y.shape)})"
            )

    @property
    def num_node_features(self) -> int:
        """
       Returns the dimensionality of the input feature list for each graph sample.

       Returns:
           int: The number of input features.
       """
        return self[0].x.size(-1)

    def get(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Read and decode a single event from the LMDB dataset.

        Args:
            idx (int): Index of the event to retrieve.

        Raises:
            MissingFieldError, DataError
        """
        if self.target_labels is None:
            self.target_labels = list(self.attrs.values())[0]["global"]["target_labels"]
        if self.include_labels is None:
            self.include_labels = list(self.attrs.values())[0]["global"]["include_labels"]

        # initialize the data and attribute dict
        data_dict = {}
        attr_dict = {}

        if not isinstance(idx, (int, np.integer)):
            raise DataError(f"Index must be int, got {type(idx).__name__!r}")

        # fetch the record from the reader
        reader = self._ensure_reader()
        try:
            data, *_ = reader[idx]
        except (IndexError, KeyError) as e:
            raise DataError(f"Failed to retrieve record at index {idx}: {e}")

        # features
        try:
            data_dict["x"] = torch.tensor(data["features"], dtype=torch.float32)
        except KeyError:
            raise MissingFieldError(f"Record at index {idx} missing 'features' field")
        if data_dict["x"].ndim == 1:
            data_dict["x"] = data_dict["x"].unsqueeze(0)  # [F] -> [1, F]

        # labels
        labels_vals = []
        for name in self.target_labels:
            if name not in data:
                raise MissingFieldError(f"Label '{name}' not found in record at index {idx}")
            labels_vals.append(data[name])
        data_dict["y"] = torch.tensor(labels_vals, dtype=torch.float32).unsqueeze(0)  # [1, L]

        # edges
        try:
            data_dict["edge_index"] = torch.tensor(data["edge_index"], dtype=torch.long)
        except KeyError:
            raise MissingFieldError(f"Record at index {idx} missing 'edge_index' field")
        try:
            data_dict["edge_attr"] = torch.tensor(data["edge_weight"], dtype=torch.float32)
        except KeyError:
            raise MissingFieldError(f"Record at index {idx} missing 'edge_weight' field")

        # normalize shapes / sanity checks
        if data_dict["edge_index"].ndim != 2:
            raise DataError(f"'edge_index' must be 2-D, got {tuple(data_dict['edge_index'].shape)} at index {idx}")
        if data_dict["edge_index"].shape[0] == 2:
            pass  # [2, E]
        elif data_dict["edge_index"].shape[1] == 2:
            data_dict["edge_index"] = data_dict["edge_index"].T.contiguous()  # [E,2] -> [2,E]
        else:
            raise DataError(f"'edge_index' must be [2, E] or [E, 2], got {tuple(data_dict['edge_index'].shape)} at index {idx}")

        if data_dict["edge_attr"].ndim == 2 and data_dict["edge_attr"].shape[1] == 1:
            data_dict["edge_attr"] = data_dict["edge_attr"].reshape(-1)
        if data_dict["edge_attr"].ndim != 1:
            raise DataError(f"'edge_attr' must be 1-D (or [E,1]), got {tuple(data_dict['edge_attr'].shape)} at index {idx}")

        if data_dict["edge_index"].shape[1] != data_dict["edge_attr"].shape[0]:
            raise DataError(
                f"edge count mismatch: edge_index has {data_dict['edge_index'].shape[1]} edges "
                f"but edge_attr has {data_dict['edge_attr'].shape[0]} at index {idx}"
            )

        # included labels (val/test only)
        if self.subset in ["validation", "test"]:
            inc_vals = []
            for name in self.include_labels:
                if name in self.target_labels:
                    continue
                if name not in data:
                    raise MissingFieldError(f"Included label '{name}' not found in record at index {idx}")
                inc_vals.append(data[name])
            if inc_vals:
                attr_dict["include_labels"] = torch.tensor(inc_vals, dtype=torch.float32).unsqueeze(0)  # [1, N_inc]

        return data_dict, attr_dict
