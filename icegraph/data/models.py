# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean


from typing import Self

from torch.utils.data import Dataset

from icegraph.converter import HDF5ToParquet
from icegraph.extractor import FeatureExtractor
from icegraph.cache import I3DirectoryCacheHandler
from icegraph.console import Console



class Data(Dataset):

    def __init__(self, data_dir: str) -> None:
        super(Data, self).__init__()
        self.data_dir = data_dir

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

    @classmethod
    def from_i3(cls, _dir, feature_extraction_config_path: str) -> Self:
        """Returns an instance of ``Data`` using an i3 file dataset. All i3 files are converted to parquet databases."""

        Console.out("Querying cache for pre-converted data...")
        converted_files = I3DirectoryCacheHandler.query(_dir, feature_extraction_config_path)
        if converted_files:
            Console.out("Cached data found!")
            return cls(converted_files)
        Console.out("No cached data found")

        # extract features to hdf5
        extractor = FeatureExtractor(_dir, feature_extraction_config_path)
        hdf5_file = extractor.extract()

        source_hash = I3DirectoryCacheHandler.get_hash(_dir, feature_extraction_config_path)
        # convert hdf5 to parquet for fast data queries
        converter = HDF5ToParquet(hdf5_file, source_hash)
        converted_files = converter.convert()

        I3DirectoryCacheHandler.register(_dir, converted_files, feature_extraction_config_path)
        return cls(converted_files)
