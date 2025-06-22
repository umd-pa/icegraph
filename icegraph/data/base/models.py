# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import Union, List
from pathlib import Path
from abc import ABC
import functools
import os
import re
from multiprocessing import get_context

import pyarrow.parquet as pq
from torch.utils.data import Dataset, DataLoader, get_worker_info
import torch
import pandas as pd
import numpy as np

from icegraph.data.converter import generate_vector_mapping, HDF5ToParquet
from icegraph.config import IGConfig
from icegraph.console import Console
from icegraph.data.base.workers import set_cache_inst, cache_event_worker
from icegraph.data.cache import IGDataCache
from icegraph.geometry import Detector

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
        target_labels (list[str]): List of target label keys to extract per event.
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
        self.event_id_columns = config.standard_id_col_config.event_id_columns
        self.dom_id_columns = config.standard_id_col_config.dom_id_columns
        self.dom_position_columns = config.standard_id_col_config.dom_position_columns

        # Load truth labels and parquet handle
        self.truth_df: pd.DataFrame = pd.read_parquet(self.data_dir / "truth.parquet")
        self.features_file: pq.ParquetFile = pq.ParquetFile(self.data_dir / "features.parquet")

        # Flags and cache
        self._truth_filtered: bool = False
        self._use_cache: bool = use_cache
        self._data_cache: IGDataCache = data_cache

        # Apply subset filtering and index labels
        self._drop_subset_indices()
        self.truth_df.reset_index(drop=True)

        self.target_labels = self._config.user_config.data.target_labels

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
        return len(self.truth_df)

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
            row = self.truth_df.iloc[idx]
            keys: dict[str, int] = {
                col: int(row[col]) for col in self.event_id_columns
            }
            labels = row[self.target_labels].to_numpy(dtype=np.float32)
            features = self._get_features_for_event(keys)

        # convert to tensors
        return torch.from_numpy(features), torch.from_numpy(labels)

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

        total = len(self)
        items = list(range(total))

        # Use fork context to inherit module globals on Unix
        ctx = get_context('fork')
        workers = min(self._config.user_config.execution.workers, os.cpu_count() or 1)
        Console.out(f"Starting multiprocessing cache fill for subset='{self.subset}' with {workers} workers.")
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

        selection_str = getattr(self._config.user_config.data.splits, self.subset)
        Console.out(f"Applying selection string for {self.subset=}: {selection_str}")

        def safe_query(df, expr):
            # find all words in the expression
            names = set(re.findall(r"[A-Za-z_]\w*", expr))
            missing = names - set(df.columns) - set(dir(__builtins__))
            if missing:
                raise KeyError(f"Unknown column(s) in query: {missing}")
            return df.query(expr)

        selected_idx = safe_query(self.truth_df, selection_str).index
        self.truth_df = self.truth_df.loc[selected_idx]
        self._truth_filtered = True

    def _read_event_batches(self, keys: dict[str, int]) -> List[pd.DataFrame]:
        """
        Read all row-groups matching a given set of keys and return them as pandas dataframes.

        Args:
            keys (str): The event identifiers to filter.

        Returns:
            List[pd.DataFrame]: Dataframes containing rows for the event.
        """
        dataframes: List[pd.DataFrame] = []
        for row_group in range(self.features_file.num_row_groups):
            table = self.features_file.read_row_group(
                row_group,
                columns=self.event_id_columns
                        + self.dom_id_columns
                        + self.features_columns
            )
            df = table.to_pandas()

            # build mask in pandas
            mask = pd.Series(True, index=df.index)
            for col, val in keys.items():
                mask &= (df[col] == val)

            sub = df.loc[mask]
            if not sub.empty:
                dataframes.append(sub)

        return dataframes

    def _get_features_for_event(self, keys: dict[str,int]) -> np.ndarray:
        """
        Retrieve DOM-level feature vectors for a given event.

        Args:
            keys (str): Event identifier string.

        Returns:
            np.ndarray: 2D array of shape (num_DOMs, num_features + 3) for the event.

        Raises:
            ValueError: If no features were found for the given event ID.
        """
        # Scan row groups for matching base_id
        batches = self._read_event_batches(keys)
        if not batches:
            raise ValueError(f"No features found for event keys {keys}")

        # concatenate all matching pieces
        full_df = pd.concat(batches, ignore_index=True)
        feat_df = full_df[self.features_columns].copy()

        detector = Detector(self._config)

        # map dom ids to dom positions
        pos_series = full_df.apply(
            lambda row: detector.get_dom_coords(
                row[self.dom_id_columns[0]], row[self.dom_id_columns[1]], row[self.dom_id_columns[2]]
            ),
            axis=1
        )
        pos_df = pd.DataFrame(
            pos_series.tolist(),
            index=full_df.index,
            columns=self.dom_position_columns
        )

        # add it to the features dataframe
        feat_df = pd.concat([feat_df, pos_df], axis=1)

        return feat_df.to_numpy(dtype="float32")
