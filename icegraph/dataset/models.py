# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import Self, Union
from pathlib import Path

from icegraph.config import Config
from torch.utils.data import Dataset

from icegraph.converter import HDF5ToParquet
from icegraph.extractor import FeatureExtractor
from icegraph.cache import I3ConversionCache
from icegraph.console import Console


__all__ = ["Data"]

class Data(Dataset):
    """
    A dataset class for loading and managing IceCube data.
    """

    def __init__(self, data_dir: Union[str, Path]) -> None:
        """
        Initialize a Data object from a given directory.

        Args:
            data_dir (Union[str, Path]): Path to the directory containing converted Parquet data.
        """
        super(Data, self).__init__()
        self.data_dir = Path(data_dir)

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        pass

    def __getitem__(self, idx):
        """
        Retrieve a sample from the dataset by index.

        Args:
            idx (int): Index of the sample.

        Returns:
            Sample corresponding to the given index.
        """
        pass

    @classmethod
    def from_config(cls, config: Config) -> Self:
        """
        Create a Data instance from a Config object. Checks for cached conversion results first.

        Args:
            config (Config): IceGraph configuration object containing user settings.

        Returns:
            Data: A new Data instance initialized from converted data.
        """
        # check the cache for a pre-converted file before running
        Console.out(f"Looking for cached conversion of: {config.user_config.input_dir}")

        # initialize the cache handler
        cache_handler = I3ConversionCache(config)

        if cached := cache_handler.query():
            Console.out(f"Cached data found: {cached}")
            return cls(cached)

        Console.out("No cached data found, running conversion")
        return cls._generate_from_config(config, cache_handler)

    @classmethod
    def _generate_from_config(cls, config: Config, cache: I3ConversionCache) -> Self:
        """
        Perform full feature extraction and conversion pipeline from configuration.

        Args:
            config (Config): IceGraph configuration object containing user settings.
            cache (I3ConversionCache): The cache handler to manage and register conversions.

        Returns:
            Data: A new Data instance built from extracted and converted data.
        """
        # extract features to hdf5
        extractor = FeatureExtractor(config)
        extracted_features = extractor.extract()

        # convert hdf5 to parquet for fast data queries
        converter = HDF5ToParquet(config, extracted_features)
        converted_files = converter.convert()

        cache.register(converted_files)
        return cls(converted_files)
