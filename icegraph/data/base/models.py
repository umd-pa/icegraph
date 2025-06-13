# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import Union
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa
from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
import numpy as np
from abc import ABC

from icegraph.data.converter import generate_vector_mapping, HDF5ToParquet
from icegraph.config import IGConfig
from icegraph.console import Console

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

    def __init__(self, data_dir: Union[str, Path], config: IGConfig) -> None:
        """
        Initialize an IGData object from a directory containing Parquet files.

        Args:
            data_dir (Union[str, Path]): Path to the directory containing 'truth.parquet' and 'features.parquet'.
            config (IGConfig): IceGraph configuration object containing user settings.

        Raises:
            NotImplementedError: If the `subset` class attribute is not defined in a subclass.
        """
        super(IGData, self).__init__()
        self.data_dir = Path(data_dir)
        self._config: IGConfig = config

        self.features_columns = list(generate_vector_mapping(config).values())

        self.truth_df: pd.DataFrame = pd.read_parquet(self.data_dir / "truth.parquet")
        self.features_file: pq.ParquetFile = pq.ParquetFile(self.data_dir / "features.parquet")

        # initialize cache attributes
        self._truth_filtered: bool = False

        # prepare truth table and mappings
        self.drop_subset_indices()

        self.truth_df.set_index('event_id', inplace=True)
        self.event_ids = list(self.truth_df.index)

        self.target_labels = self._config.user_config.target_labels
        self.label_map = self.truth_df[self.target_labels].to_dict()

        # preload metadata to speed things up later
        self.metadata = self.features_file.metadata

        # verify self.subset has been specified
        if not self.subset:
            raise NotImplementedError(
                f"Subclasses of IGData must define the class attribute IGData.subset as one of ['train', 'validation', 'test']."
            )

    def __len__(self) -> int:
        """
        Return the number of events in the dataset.

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
        event_id = self.event_ids[idx]

        labels = []
        for label in self.label_map:
            labels.append(self.label_map[label][event_id])

        feature_array = self._get_features_for_event(event_id)

        return torch.tensor(feature_array), torch.tensor(labels)

    @property
    def dataloader(self, **kwargs) -> DataLoader:
        """
        Returns a PyTorch DataLoader for this dataset instance.

        Args:
            **kwargs: Arguments to pass to torch.utils.data.DataLoader.

        Returns:
            DataLoader: PyTorch DataLoader instance.
        """
        return DataLoader(self, **kwargs)

    def drop_subset_indices(self) -> None:
        """
        Applies a selection filter to keep only the subset of truth_df that matches the config-defined criteria.

        This is done once during initialization. It filters `self.truth_df` in-place by parsing event numbers
        from the 'event_id' field and applying a selection string defined in the config.
        """
        if self._truth_filtered:
            return  # Already applied

        selection_str = getattr(self._config.user_config.selection, self.subset)
        Console.out(f"Using selection string for {self.subset=}: {selection_str}", severity=1)

        # Extract 'Event' safely from 'event_id'
        selected_events = self.truth_df["event_id"].str.extract(r"Event=(\d+)")[0]
        selected_events = selected_events.dropna().astype(int)

        valid_indices = selected_events.index

        selected_indices = (
            self.truth_df
            .loc[valid_indices]
            .assign(Event=selected_events)
            .query(selection_str)
            .index
        )

        self.truth_df = self.truth_df.loc[selected_indices]
        self._truth_filtered = True

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
        matching_batches = []

        for rg in range(self.features_file.num_row_groups):
            row_group_table = self.features_file.read_row_group(
                rg, columns=["event_id", "dom_id"] + self.features_columns
            )
            ids = row_group_table.column("event_id").to_pylist()
            mask = [i == event_id for i in ids]

            # continue loop if none on this iteration
            if not any(mask):
                continue

            # Filter relevant rows
            mask_array = pa.array(mask)
            batch = row_group_table.filter(mask_array)
            matching_batches.append(batch)

        if not matching_batches:
            raise ValueError(f"No features found for event {event_id}")

        # Concatenate all matching batches
        all_rows = pa.concat_tables(matching_batches)
        df = all_rows.to_pandas().drop(columns=['event_id', "dom_id"])  # Drop ID, keep only features
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

        labels = []
        for label in self.label_map:
            labels.append(self.label_map[label][event_id])

        feature_array = self._get_features_for_event(event_id)

        dom_ids = np.array(self.features_file.read(columns=["dom_id"]).to_pandas())
        unpacked_dom_ids = []
        for _id in dom_ids:
            unpacked_dom_ids.append(self._unpack_id(_id[0]))

        return np.array(feature_array), np.array(labels), np.array(unpacked_dom_ids)

    @staticmethod
    def _unpack_id(_id: str) -> list[int]:
        """
        Unpacks a DOM ID string (formatted as 'key=val|key=val|...') into a list of integers.

        Args:
            _id (str): Packed DOM ID string.

        Returns:
            list[int]: List of extracted integer values from the ID string.
        """
        # split the id
        split_id = _id.split("|")

        # parse and generate a list
        unpacked_id = [int(entry.split("=")[1]) for entry in split_id]

        return unpacked_id
