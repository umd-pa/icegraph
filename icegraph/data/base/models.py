# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import Union, List
from pathlib import Path
from abc import ABC
import functools
import os
from multiprocessing import get_context

import pyarrow.parquet as pq
import pyarrow as pa
from icegraph.data.cache import IGDataCache
from torch.utils.data import Dataset, DataLoader, get_worker_info
import torch
import pandas as pd
import numpy as np

from icegraph.data.converter import generate_vector_mapping, HDF5ToParquet
from icegraph.config import IGConfig
from icegraph.console import Console
from icegraph.data.base.workers import set_cache_inst, cache_event_worker

__all__ = ["IGData"]


class IGData(Dataset, ABC):
    """
    The base dataset class for loading and managing IceCube data stored in Parquet format.

    This class handles the truth table, feature loading, and optional selection filtering
    for training, validation, or test subsets. Subclasses must set the class attribute `subset`
    to one of: "train", "validation", or "test".

    Attributes:
        data_dir (Path): Path to the directory containing the Parquet files.
        _config (IGConfig): Configuration object with user-defined settings.
        features_columns (list[str]): List of feature column names to extract.
        truth_df (pd.DataFrame): DataFrame storing the truth labels indexed by event_id.
        features_file (pq.ParquetFile): Parquet file storing DOM-level features.
        event_ids (list[str]): List of selected event IDs after applying filtering.
        target_labels (list[str]): List of target label keys to extract per event.
        label_map (dict): Mapping from event_id to target labels.
        metadata (pa.Metadata): Cached metadata from the feature file.
        _truth_filtered (bool): Flag to ensure subset filtering is applied only once.
    """

    subset: str | None = None

    dataloader = property(
        lambda self: functools.partial(DataLoader, self),
        doc="A convenience property that returns a partially-applied torch DataLoader constructor for this dataset."
    )

    def __init__(
        self,
        data_dir: Union[str, Path],
        config: IGConfig,
        data_cache: IGDataCache,
        *,
        use_cache=True
    ) -> None:
        """
        Initialize an IGData object from a directory containing Parquet files.

        Args:
            data_dir (Union[str, Path]): Path to the directory containing 'truth.parquet' and 'features.parquet'.
            config (IGConfig): IceGraph configuration object containing user settings.
            data_cache (IGDataCache): Cache handler for on-disk feature caching.
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self._config: IGConfig = config

        # Determine which features to load
        self.features_columns = list(generate_vector_mapping(config).values())

        # Load truth labels and parquet handle
        self.truth_df: pd.DataFrame = pd.read_parquet(self.data_dir / "truth.parquet")
        self.features_file: pq.ParquetFile = pq.ParquetFile(self.data_dir / "features.parquet")

        # Flags and cache
        self._truth_filtered: bool = False
        self._use_cache: bool = use_cache
        self._data_cache: IGDataCache = data_cache

        # Apply subset filtering and index labels
        self._drop_subset_indices()
        self.truth_df.set_index('event_id', inplace=True)
        self.event_ids = list(self.truth_df.index)

        self.target_labels = self._config.user_config.target_labels
        self.label_map = self.truth_df[self.target_labels].to_dict()

        # Preload metadata for faster access
        self.metadata = self.features_file.metadata

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

    def __len__(self) -> int:
        """
        Return the number of events in the subset.

        Returns:
            int: Number of events.
        """
        return len(self.event_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve a single sample by index.

        Args:
            idx (int): Index of the event.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple of (features, labels) for the selected event.
        """
        if self._use_cache and self._data_cache is not None:
            features, labels = self._data_cache.query(self, idx)
        else:
            event_id = self.event_ids[idx]
            labels = np.array([self.label_map[label][event_id] for label in self.target_labels])
            features = self._get_features_for_event(event_id)
        return torch.tensor(features), torch.tensor(labels)

    def setup_cache(self) -> None:
        """
        Warm per-event cache on the main process if required.
        """
        if self._use_cache:
            worker_info = get_worker_info()
            if worker_info is None and self._data_cache.build_required:
                # main process: build the cache once
                self._populate_cache()

    def _populate_cache(self) -> None:
        """
        Populate the on-disk cache in parallel using multiprocessing.

        This distributes event-level cache writes across multiple processes.
        Requires a module-level worker to be picklable.
        """
        # Register this instance for worker access
        set_cache_inst(self)

        items = list(enumerate(self.event_ids))
        total = len(items)

        # Use fork context to inherit module globals on Unix
        ctx = get_context('fork')
        workers = min(self._config.user_config.multiprocessing.workers, os.cpu_count())
        Console.out(f"Starting multiprocessing pool with {workers} workers.")
        with ctx.Pool(processes=workers) as pool:
            for _ in Console.progress_bar(
                pool.imap_unordered(cache_event_worker, items),
                total=total
            ):
                pass

    def _drop_subset_indices(self) -> None:
        """
        Applies a selection filter to keep only the subset of truth_df that matches the config-defined criteria.

        This is done once during initialization. It filters `self.truth_df` in-place by parsing event numbers
        from the 'event_id' field and applying a selection string defined in the config.
        """
        if self._truth_filtered:
            return

        selection_str = getattr(self._config.user_config.selection, self.subset)
        Console.out(f"Applying selection string for {self.subset=}: {selection_str}", severity=1)

        events = self.truth_df["event_id"].str.extract(r"Event=(\d+)")[0].astype(int)
        selected = self.truth_df.assign(Event=events).query(selection_str).index
        self.truth_df = self.truth_df.loc[selected]
        self._truth_filtered = True

    def _read_event_batches(self, event_id: str) -> List[pa.Table]:
        """
        Read all row-groups matching a given event_id and return them as Arrow Tables.

        Args:
            event_id (str): The event identifier to filter.

        Returns:
            List[pa.Table]: Arrow Tables containing rows for the event.
        """
        batches: List[pa.Table] = []
        for rg in range(self.features_file.num_row_groups):
            table = self.features_file.read_row_group(
                rg, columns=["event_id", "dom_id"] + self.features_columns
            )
            ids = table.column("event_id").to_pylist()
            mask = [i == event_id for i in ids]
            if not any(mask):
                continue
            batches.append(table.filter(pa.array(mask)))
        return batches

    def _get_features_for_event(self, event_id: str) -> np.ndarray:
        """
        Retrieve DOM-level feature vectors for a given event.

        Args:
            event_id (str): Event identifier string.

        Returns:
            np.ndarray: 2D array of shape (num_DOMs, num_features) for the event.

        Raises:
            ValueError: If no features were found for the given event ID.
        """
        # Scan row groups for matching base_id
        batches = self._read_event_batches(event_id)
        if not batches:
            raise ValueError(f"No features found for event {event_id}")

        # Concatenate all matching batches and drop ID columns
        all_rows = pa.concat_tables(batches)
        df = all_rows.to_pandas().drop(columns=['event_id', 'dom_id'])
        return df.to_numpy(dtype='float32')

    def get_with_dom_id(self, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Retrieve a sample by index, along with DOM IDs.

        This is useful for visualization or analysis that requires spatial DOM context.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]:
                - Feature array (num_DOMs, num_features)
                - Labels array (num_labels,)
                - DOM ID array (num_DOMs, 2) with [string, om]
        """
        event_id = self.event_ids[idx]

        # --- features & labels (use cache if available) ---
        if self._use_cache and self._data_cache is not None:
            features, labels = self._data_cache.query(self, idx)
        else:
            labels = np.array([self.label_map[label][event_id] for label in self.target_labels])
            features = self._get_features_for_event(event_id)

        # --- dom_id extraction ---
        batches = self._read_event_batches(event_id)
        if not batches:
            raise ValueError(f"No DOMs found for event {event_id}")

        all_rows = pa.concat_tables(batches)
        df_ids = all_rows.column("dom_id").to_pandas()
        dom_ids = np.stack(df_ids.apply(self._unpack_id).to_list(), axis=0)

        return features, labels, dom_ids

    @staticmethod
    def _unpack_id(_id: str) -> list[int]:
        """
        Unpacks a DOM ID string (formatted as 'key=val|key=val|...') into a list of integers.

        Args:
            _id (str): Packed DOM ID string.

        Returns:
            list[int]: List of extracted integer values from the ID string.
        """
        return [int(x.split("=")[1]) for x in _id.split("|")]
