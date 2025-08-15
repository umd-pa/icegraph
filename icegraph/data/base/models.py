# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import Union, Optional, Literal, Sequence, ClassVar, Dict, Any, List
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
from icegraph.data.splitter import SplitMapBuilder
from icegraph.data.base.exceptions import EmptyDatasetError, DataError, MissingFieldError
from icegraph.data.readers import LMDBConfiguredShardReader, LMDBReader
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
    _reader:            ClassVar[Optional[LMDBConfiguredShardReader]]                       = None
    _source:            ClassVar[Optional[Union[str, Path, Sequence[Union[str, Path]]]]]    = None
    _map_dataframe:     ClassVar[Optional[pd.DataFrame]]                                    = None

    # reader process id for forks
    _reader_pid:        ClassVar[Optional[int]]                                             = None

    # dataset metadata
    metadata:           ClassVar[Optional[Dict[str, Any]]]                                  = None

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
        if cls._source is None or cls._map_dataframe is None:
            raise DataError("Please run IGData.configure() before instantiating subclasses.")

        super().__init__()

        # build the reader
        cls._build_reader()

        # grab global config
        self._config: IGConfig = IGConfig.get()

        # get the key list
        split_int = SplitMapBuilder.SPLIT_INT_MAP[self.subset]
        map_df = cls._map_dataframe
        self.keys = map_df[map_df["split"] == split_int]["index"].tolist()

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
        features, labels, edge_index, edge_weight = self.get(idx)
        return PyGData(x=features, y=labels, edge_index=edge_index, edge_attr=edge_weight)

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
            cls._reader = LMDBConfiguredShardReader()

    @classmethod
    def configure(cls, source: Union[str, Path, Sequence[Union[str, Path]]], map_file: Union[str, Path]) -> None:
        """
        Set the dataset configurations.

        Args:
            source (Union[str, Path, Sequence[Union[str, Path]]]): Path to the input file(s) (LMDB) or a directory.
            map_file (Union[str, Path]): Path to the split mapping file generated by `SplitMapBuilder`
        """
        if cls._source is None:
            cls._source = source
        if cls._map_dataframe is None:
            cls._map_dataframe = LMDBReader(map_file).to_pandas().sort_values(by="index").reset_index(drop=True)

        # configure the reader class
        LMDBConfiguredShardReader.configure(
            source=cls._source,
            map_df=cls._map_dataframe,
            max_open_envs=4,
            clean=True
        )

        # load metadata
        cls.metadata = LMDBConfiguredShardReader.metadata()

        # verify config hash
        config = cls.metadata["config"]
        config_hash = cls.metadata["CBOR_canonical_blake2b"]

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
        else:
            raise DataError(
                f"Expected label tensor to be 0D or 1D, got {y.ndim}D (shape={tuple(y.shape)})"
            )

    @property
    def num_node_features(self):
        """
       Returns the dimensionality of the input feature list for each graph sample.

       Returns:
           int: The number of input features.
       """
        return self[0].x.size(-1)

    def get(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        # validate index
        if not isinstance(idx, int):
            raise DataError(f"Index must be int, got {type(idx).__name__!r}")

        # fetch record via reader
        try:
            data, *_ = type(self)._reader[idx]  # dict unpacked from msgpack (string keys)
        except (IndexError, KeyError) as e:
            raise DataError(f"Failed to retrieve record at index {idx}: {e}")

        # normalize

        # --- features ---
        try:
            features_np = np.array(data["features"], dtype=np.float32, copy=True)
        except KeyError:
            raise MissingFieldError(f"Record at index {idx} missing 'features' field")

        # --- labels ---
        labels_vals = []
        for name in self.metadata["target_labels"]:
            if name not in data:
                raise MissingFieldError(f"Label '{name}' not found in record at index {idx}")
            labels_vals.append(data[name])
        labels_np = np.asarray(labels_vals, dtype=np.float32)

        # --- edges ---
        try:
            ei_list = data["edge_index"]
        except KeyError:
            raise MissingFieldError(f"Record at index {idx} missing 'edge_index' field")
        try:
            ew_list = data["edge_weight"]
        except KeyError:
            raise MissingFieldError(f"Record at index {idx} missing 'edge_weight' field")

        ei_np = np.asarray(ei_list, dtype=np.int64)
        ew_np = np.asarray(ew_list, dtype=np.float32)

        # sanity checks
        if ei_np.ndim != 2 or ei_np.shape[0] != 2:
            raise DataError(f"'edge_index' must be shape [2, E], got {ei_np.shape} at index {idx}")
        if ew_np.ndim != 1:
            raise DataError(f"'edge_weight' must be 1-D, got {ew_np.shape} at index {idx}")
        if ei_np.shape[1] != ew_np.shape[0]:
            raise DataError(
                f"edge count mismatch: edge_index has {ei_np.shape[1]} edges "
                f"but edge_weight has {ew_np.shape[0]} at index {idx}"
            )

        # to tensors
        features = torch.from_numpy(features_np)
        labels = torch.from_numpy(labels_np)
        edge_index = torch.from_numpy(ei_np).long()
        edge_weight = torch.from_numpy(ew_np)

        return features, labels, edge_index, edge_weight
