# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import Self, Union
from pathlib import Path

import pyarrow.parquet as pq
from torch.utils.data import Dataset
import torch

from icegraph.converter import HDF5ToParquet
from icegraph.extractor import FeatureExtractor
from icegraph.cache import IGConversionCache
from icegraph.console import Console
from icegraph.converter import MLSuiteVectorMapping
from icegraph.config import IGConfig


__all__ = ["IGData"]

class IGData(Dataset):
    """
    A dataset class for loading and managing IceCube data.
    """

    def __init__(self, data_dir: Union[str, Path], config: IGConfig) -> None:
        """
        Initialize a Data object from a given directory.

        Args:
            config (IGConfig): IceGraph configuration object containing user settings.
            data_dir (Union[str, Path]): Path to the directory containing converted Parquet data.
        """
        super(IGData, self).__init__()
        self.data_dir = Path(data_dir)

        self.features_columns = list(MLSuiteVectorMapping(config).get_mapping().values())

        self.data_features = pq.ParquetFile(self.data_dir / "features.parquet")
        self.data_truth = pq.ParquetFile(self.data_dir / "truth.parquet")

        print(self.data_features.metadata.num_rows, self.data_truth.metadata.num_rows)

        assert self.data_features.metadata.num_rows == self.data_truth.metadata.num_rows

        self.length = self.data_features.metadata.num_rows

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return self.length

    def __getitem__(self, idx):
        """
        Retrieve a sample from the dataset by index.

        Args:
            idx (int): Index of the sample.

        Returns:
            Sample corresponding to the given index.
        """
        row_group_index, index_in_group = self._get_row_group_and_offset(idx)

        # Read row group (can optimize by caching small ones)
        features_table = self.data_features.read_row_group(row_group_index, columns=self.features_columns)
        truth_table = self.data_truth.read_row_group(row_group_index, columns=["value"])

        # Extract the row
        features_row = features_table.slice(index_in_group, 1).to_pydict()
        label_row = truth_table.slice(index_in_group, 1).to_pydict()

        features = torch.tensor([features_row[col][0] for col in self.features_columns], dtype=torch.float32)
        label = torch.tensor(label_row["value"][0], dtype=torch.float32)

        return features, label

    def _get_row_group_and_offset(self, idx):
        total = 0
        for i in range(self.data_features.num_row_groups):
            rg_size = self.data_features.metadata.row_group(i).num_rows
            if idx < total + rg_size:
                return i, idx - total
            total += rg_size
        raise IndexError("Index out of bounds")

    @classmethod
    def from_config(cls, config: IGConfig) -> Self:
        """
        Create a Data instance from a Config object. Checks for cached conversion results first.

        Args:
            config (IGConfig): IceGraph configuration object containing user settings.

        Returns:
            IGData: A new Data instance initialized from converted data.
        """
        # check the cache for a pre-converted file before running
        Console.out(f"Looking for cached conversion of: {config.user_config.input_dir}")

        # initialize the cache handler
        cache_handler = IGConversionCache(config)

        if cached := cache_handler.query():
            Console.out(f"Cached data found: {cached}") 
            return cls(cached, config)

        Console.out("No cached data found, running conversion")
        return cls._generate_from_config(config, cache_handler)

    @classmethod
    def _generate_from_config(cls, config: IGConfig, cache: IGConversionCache) -> Self:
        """
        Perform full feature extraction and conversion pipeline from configuration.

        Args:
            config (IGConfig): IceGraph configuration object containing user settings.
            cache (IGConversionCache): The cache handler to manage and register conversions.

        Returns:
            IGData: A new Data instance built from extracted and converted data.
        """
        # extract features to hdf5
        extractor = FeatureExtractor(config)
        extracted_features = extractor.extract()

        # convert hdf5 to parquet for fast data queries
        converter = HDF5ToParquet(config, extracted_features)
        converted_files = converter.convert()

        cache.register(converted_files)
        return cls(converted_files, config)
