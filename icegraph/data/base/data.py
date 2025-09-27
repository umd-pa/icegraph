# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

import os
from typing import Union, Optional, Literal, Sequence, ClassVar, Dict, Any, List, Tuple, TypeVar, Callable, cast
from pathlib import Path
from abc import ABC
import functools
import itertools
from contextlib import ExitStack

from torch.utils.data import Dataset
import torch_geometric as pyg
from torch_geometric.data import Data as PyGData
import torch
import numpy as np
from numpy.typing import NDArray

from icegraph.config import IGConfig
from icegraph.data.base.exceptions import NotConfiguredError, DataError, MissingFieldError
from icegraph.data.readers import LMDBDatasetShardReader
from icegraph.utils import stable_hash_cbor

__all__ = ["IGData"]


F = TypeVar("F", bound=Callable[..., Any])

def requires_config(func):
    @functools.wraps(func)
    def w(self, *a, **k):
        if getattr(self, "_source", None) is None:
            raise NotConfiguredError("IGData must be configured before use.")
        return func(self, *a, **k)
    return w


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
    _source:            ClassVar[Optional[Union[str, Path, Sequence[Union[str, Path]]]]]    = None

    # dataset attributes
    attrs:              ClassVar[Optional[Dict[int, Dict[str, Dict[str, Any]]]]]            = None
    include_labels:     ClassVar[Optional[List[str]]]                                       = None
    target_labels:      ClassVar[Optional[List[str]]]                                       = None

    dataloader = property(
        lambda self: functools.partial(pyg.loader.DataLoader, self),
        doc="A convenience property that returns a partially-applied torch geometric DataLoader constructor for this dataset."
    )

    @requires_config
    def __init__(self) -> None:
        """
        Initialize an IGData object from an LMDB file.
        """
        super().__init__()

        self._stack:        Optional[ExitStack]                 = None
        self._proc_pid:     Optional[int]                       = None
        self._reader:       Optional[LMDBDatasetShardReader]    = None

        # grab global config
        self._config: IGConfig = IGConfig.get()

        # get the key list
        self._keys: Optional[np.ndarray] = None

    def __init_subclass__(cls, **kwargs) -> None:
        """
        Validate that subclasses of IGData define a proper `subset` attribute.

        Ensures that any subclass sets the class-level `subset` to one of
        the allowed split names ("train", "validation", or "test"), and that
        it consists only of alphabetic characters.

        Raises:
            DataError: If `subset` is not defined on the subclass or if `subset` is not an alphabetic string.
        """
        super().__init_subclass__(**kwargs)
        if cls.subset is None:
            raise DataError(f"{cls.__name__}.subset must be set to 'train'|'validation'|'test'")
        if not isinstance(cls.subset, str) or not cls.subset.isalpha():
            raise DataError(f"`subset` on {cls.__name__!r} must be alphabetic string, got {cls.subset!r}")

    @requires_config
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

    @requires_config
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

    def __getstate__(self):
        # Instance base state
        s = self.__dict__.copy()

        # Drop per-process / unpicklables
        s["_stack"]     = None
        s["_reader"]    = None
        s["_finalizer"] = None
        s["_proc_pid"]  = None

        # If your IGConfig can hold loggers/locks, drop & reacquire in worker
        s["_config"] = None

        # Snapshot IGData *class-level* config so workers see it
        cls = type(self)
        s["_ig_class_snapshot"] = {
            "_source":        cls._source,
            "attrs":          cls.attrs,
            "include_labels": cls.include_labels,
            "target_labels":  cls.target_labels,
            "subset":         cls.subset,  # for completeness
        }

        # Snapshot LMDB reader *class-level* config so new readers work in workers
        s["_reader_snapshot"] = {
            "paths":         tuple(str(p) for p in (LMDBDatasetShardReader._lmdb_paths or ())),
            "max_open_envs": LMDBDatasetShardReader._max_open_envs,
            "index":         LMDBDatasetShardReader._index_arr,  # numpy is picklable
        }
        return s

    def __setstate__(self, s):
        # Restore the simple instance dict
        self.__dict__.update(s)

        # Recreate per-process handles lazily
        self._stack = None
        self._reader = None
        self._finalizer = None
        self._proc_pid = None

        # Reacquire IGConfig if needed
        if self._config is None:
            self._config = IGConfig.get()

        # Restore IGData *class-level* config in the worker
        cls = type(self)
        snap = s.get("_ig_class_snapshot", {})
        if snap:
            cls._source        = snap.get("_source",        cls._source)
            cls.attrs          = snap.get("attrs",          cls.attrs)
            cls.include_labels = snap.get("include_labels", cls.include_labels)
            cls.target_labels  = snap.get("target_labels",  cls.target_labels)
            # cls.subset is static per subclass, no change needed

        # Restore LMDB reader *class-level* config in the worker
        rs = s.get("_reader_snapshot", {})
        if rs:
            paths = rs.get("paths")
            LMDBDatasetShardReader._lmdb_paths    = tuple(Path(p) for p in paths) if paths else None
            LMDBDatasetShardReader._max_open_envs = rs.get("max_open_envs", 4)
            LMDBDatasetShardReader._index_arr     = rs.get("index", None)

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
        cls = type(self)

        # split keys are small integers (0,1,2), so uint8 is safe and more compact
        splitmap = np.fromiter(
            itertools.chain.from_iterable(attr["allocation"]["splitmap"] for attr in cls.attrs.values()),
            dtype=np.uint8
        )

        return splitmap

    @requires_config
    def _ensure_reader(self):
        """Load the dataset shard reader, ensures a unique one for each process."""
        pid = os.getpid()
        if self._reader is not None and pid == self._proc_pid:
            return self._reader

        # close old stack if any
        if self._stack is not None:
            try:
                self._stack.close()
            finally:
                self._stack = None
                self._reader = None

        # create per-process reader
        self._proc_pid = pid
        self._stack = ExitStack()
        self._reader = self._stack.enter_context(LMDBDatasetShardReader())

        return self._reader

    def close(self) -> None:
        """Close the instance."""
        if self._stack is not None:
            try:
                self._stack.close()
            finally:
                self._stack = None
                self._reader = None

    @property
    @requires_config
    def keys(self) -> NDArray:
        """
        Load the filtered-by-subset key list.

        Returns:
            NDArray: An array of keys.
        """
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

        # cache target an included labels for quick access on hot paths
        cls.include_labels = cls.attrs[0]["global"]["include_labels"]
        cls.target_labels = cls.attrs[0]["global"]["target_labels"]

        # verify config hash
        config = cls.attrs[0]["global"]["config"]
        config_hash = cls.attrs[0]["global"]["config_hash"]

        if config_hash != stable_hash_cbor(config):
            raise RuntimeError("Source config hash does not match expected hash. One or more files may be corrupted.")

    @property
    @requires_config
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
    @requires_config
    def num_node_features(self) -> int:
        """
       Returns the dimensionality of the input feature list for each graph sample.

       Returns:
           int: The number of input features.
       """
        return self[0].x.size(-1)

    @requires_config
    def get(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Read and decode a single event from the LMDB dataset.

        Args:
            idx (int): Index of the event to retrieve.

        Raises:
            MissingFieldError, DataError
        """
        cls = type(self)

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

        # --- features ---
        try:
            data_dict["x"] = torch.tensor(data["features"], dtype=torch.float32)
        except KeyError:
            raise MissingFieldError(f"Record at index {idx} missing 'features' field")
        if data_dict["x"].ndim == 1:
            data_dict["x"] = data_dict["x"].unsqueeze(0)  # [F] -> [1, F]

        # --- labels ---
        labels_vals = []
        for name in cls.target_labels:
            if name not in data:
                raise MissingFieldError(f"Label '{name}' not found in record at index {idx}")
            labels_vals.append(data[name])
        data_dict["y"] = torch.tensor(labels_vals, dtype=torch.float32).unsqueeze(0)  # [1, L]

        # --- edges ---
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

        # --- included labels (val/test only) ---
        if self.subset in ["validation", "test"]:
            inc_vals = []
            for name in cls.include_labels:
                if name in cls.target_labels:
                    continue
                if name not in data:
                    raise MissingFieldError(f"Included label '{name}' not found in record at index {idx}")
                inc_vals.append(data[name])
            if inc_vals:
                attr_dict["include_labels"] = torch.tensor(inc_vals, dtype=torch.float32).unsqueeze(0)  # [1, N_inc]

        return data_dict, attr_dict
