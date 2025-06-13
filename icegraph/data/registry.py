# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from icegraph.console import Console
from icegraph.data.cache import IGConversionCache
from icegraph.data.converter import HDF5ToParquet
from icegraph.data.extractor import FeatureExtractor
from icegraph.config import IGConfig
from icegraph.data import TrainingDataset, ValidationDataset, TestDataset

from pathlib import Path
from typing import Self, Type, TYPE_CHECKING


class DatasetRegistry:
    """
    A container class for managing access to training, validation, and test datasets.

    This class handles loading and conversion of raw input data into feature-ready
    Parquet format, applies caching, and wraps the resulting dataset objects for
    convenient access.

    Attributes:
        _train_dataset (TrainingDataset): The training dataset instance.
        _validation_dataset (ValidationDataset): The validation dataset instance.
        _test_dataset (TestDataset): The test dataset instance.
    """

    _dataset_specs: dict[str, tuple[str, Type]] = {
        "training_dataset": ("train", TrainingDataset),
        "validation_dataset": ("validation", ValidationDataset),
        "testing_dataset": ("test", TestDataset),
    }

    # --- Static property stubs for type checking and autocompletion ---
    if TYPE_CHECKING:
        @property
        def training_dataset(self) -> TrainingDataset: ...

        @property
        def validation_dataset(self) -> ValidationDataset: ...

        @property
        def testing_dataset(self) -> TestDataset: ...

    def __init__(
            self,
            train_dataset: TrainingDataset,
            validation_dataset: ValidationDataset,
            test_dataset: TestDataset
    ) -> None:
        """
        Initialize the DatasetRegistry with training, validation, and test datasets.

        Args:
            train_dataset (TrainingDataset): The training dataset.
            validation_dataset (ValidationDataset): The validation dataset.
            test_dataset (TestDataset): The test dataset.

        Raises:
            AssertionError: If the subsets of the provided datasets do not match their expected roles.
        """
        self._train_dataset = train_dataset
        self._validation_dataset = validation_dataset
        self._test_dataset = test_dataset

        # verify the datasets were passed in the correct order
        assert self._train_dataset.subset == "train"
        assert self._validation_dataset.subset == "validation"
        assert self._test_dataset.subset == "test"

    @classmethod
    def from_config(cls, config: IGConfig) -> Self:
        """
        Factory method to construct a DatasetRegistry from a configuration.

        Checks for a cached conversion; if none is found, it triggers full feature
        extraction and conversion from raw input data.

        Args:
            config (IGConfig): IceGraph configuration object containing user settings.

        Returns:
            DatasetRegistry: A fully-initialized registry containing training, validation, and test datasets.
        """
        # check the cache for a pre-converted file before running
        Console.out(f"Looking for cached conversion of: {config.user_config.input_dir}")

        # initialize the cache handler
        cache_handler = IGConversionCache(config)

        if cached := cache_handler.query():
            Console.out(f"Cached data found: {cached}")
            data = cached
        else:
            Console.out("No cached data found, running conversion")
            data = cls._generate_from_config(config, cache_handler)

        Console.out(f"Constructing dataset registry...")
        return cls(TrainingDataset(data, config), ValidationDataset(data, config), TestDataset(data, config))

    @classmethod
    def _generate_from_config(cls, config: IGConfig, cache: IGConversionCache) -> Path:
        """
        Perform feature extraction and convert the resulting HDF5 file into Parquet format.

        This is called only when no cached data is available.

        Args:
            config (IGConfig): IceGraph configuration object containing user settings.
            cache (IGConversionCache): The cache handler to manage and register the conversion.

        Returns:
            Path: The path to the converted Parquet dataset directory.
        """
        # extract features to HDF5
        extractor = FeatureExtractor(config)
        extracted_file = extractor.extract()

        # convert HDF5 to Parquet for fast data queries
        converter = HDF5ToParquet(config, extracted_file)
        converted_files = converter.convert()

        # cache the result for future reuse
        cache.register(converted_files)
        return converted_files


# --- Dynamically define accessors for splits ---
def _make_dataset_property(attr_name: str, subset_name: str, dataset_cls: Type) -> property:
    """
    Create a property accessor for a dataset corresponding to a specific data split.

    This function returns a @property that retrieves an internal attribute like
    `self._train_dataset`, `self._validation_dataset`, or `self._test_dataset` based on
    the naming convention defined in _dataset_specs.

    Args:
        attr_name (str): Name of the public property (e.g., "training_dataset").
        subset_name (str): The split name used in the internal attribute (e.g., "train").
        dataset_cls (Type): The class of the dataset (e.g., TrainingDataset).

    Returns:
        property: A dynamically constructed @property for accessing the specified dataset.
    """
    def getter(self):
        return getattr(self, f"_{subset_name}_dataset")

    getter.__name__ = attr_name
    getter.__doc__ = f"""
    Accessor for the {subset_name} dataset.

    Returns:
        {dataset_cls.__name__}: The dataset corresponding to the '{subset_name}' split.
    """
    return property(getter)

# Create a property for each dataset type (train, validation, test)
# using naming rules from the _dataset_specs mapping.
for public_name, (subset_name, dataset_cls) in DatasetRegistry._dataset_specs.items():
    setattr(DatasetRegistry, public_name, _make_dataset_property(public_name, subset_name, dataset_cls))

