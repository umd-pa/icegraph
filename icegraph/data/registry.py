# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from pathlib import Path
from typing import Self, Type, TYPE_CHECKING

from icegraph.console import Console
from icegraph.data.cache import IGConverterCache, IGDataCache
from icegraph.data.converter import HDF5ToParquet
from icegraph.data.extractor import FeatureExtractor
from icegraph.config import IGConfig
from icegraph.data import TrainingDataset, ValidationDataset, TestDataset

__all__ = ["DatasetRegistry"]


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
        "test_dataset": ("test", TestDataset),
    }

    # --- Static property stubs for type checking and autocompletion ---
    if TYPE_CHECKING:
        @property
        def training_dataset(self) -> TrainingDataset: ...

        @property
        def validation_dataset(self) -> ValidationDataset: ...

        @property
        def test_dataset(self) -> TestDataset: ...

    def __init__(
            self,
            train_dataset: TrainingDataset,
            validation_dataset: ValidationDataset,
            test_dataset: TestDataset,
            data_cache: IGDataCache
    ) -> None:
        """
        Initialize the DatasetRegistry with training, validation, and test datasets.

        Args:
            train_dataset (TrainingDataset): The training dataset.
            validation_dataset (ValidationDataset): The validation dataset.
            test_dataset (TestDataset): The test dataset.
            data_cache (IGDataCache): IceGraph data cache handler.
        """
        self._train_dataset = train_dataset
        self._validation_dataset = validation_dataset
        self._test_dataset = test_dataset

        self._datasets = [self._train_dataset, self._validation_dataset, self._test_dataset]

        self.cache: IGDataCache = data_cache

        # verify the datasets were passed in the correct order
        if self._train_dataset.subset != "train":
            raise ValueError("Expected train_dataset.subset == 'train'")
        if self._validation_dataset.subset != "validation":
            raise ValueError("Expected validation_dataset.subset == 'validation'")
        if self._test_dataset.subset != "test":
            raise ValueError("Expected test_dataset.subset == 'test'")

    def __len__(self) -> int:
        """
        Return the number of events in the full dataset.

        Returns:
            int: Number of events.
        """
        return sum(map(len, self._datasets))

    @classmethod
    def from_config(cls, config: IGConfig, use_cache=True) -> Self:
        """
        Factory method to construct a DatasetRegistry from a configuration.

        Checks for a cached conversion; if none is found, it triggers full feature
        extraction and conversion from raw input data.

        Args:
            config (IGConfig): IceGraph configuration object containing user settings.
            use_cache (bool): Pre-cache all data to speed up training, defaults to True.

        Returns:
            DatasetRegistry: A fully-initialized registry containing training, validation, and test datasets.
        """
        # check the cache for a pre-converted file before running
        Console.out(f"Looking for cached conversion of: {config.user_config.input_dir}")

        # initialize the converter cache handler
        converter_cache = IGConverterCache(config)

        if cached := converter_cache.query():
            Console.out(f"Cached data found: {cached}")
            data = cached
        else:
            Console.out("No cached data found, running conversion", severity=2)
            data = cls._generate_from_config(config, converter_cache)

        # setup data cache handler
        data_cache = IGDataCache(config)

        Console.out(f"Constructing dataset registry...")
        registry = cls(
            TrainingDataset(data, config, data_cache, use_cache=use_cache),
            ValidationDataset(data, config, data_cache, use_cache=use_cache),
            TestDataset(data, config, data_cache, use_cache=use_cache),
            data_cache
        )

        if use_cache:
            registry._build_cache()

        return registry

    def _build_cache(self) -> bool:
        """
        Ensure the on‑disk data cache is both present and valid, rebuilding it
        if necessary, and then pre‑populate each Dataset’s own cache for fast lookups.

        Returns:
            bool: True if cache was already valid; False if it was reset and rebuilt.
        """
        # Acquire lock for atomic cache validation and reset
        with self.cache.lock:
            valid_cache = self.cache.check(len(self))
            if not valid_cache:
                Console.out("Cache invalid: resetting global cache.", severity=2)
                self.cache.reset()

        # If we performed a reset, build per-split caches
        if not valid_cache:
            Console.out("Building per-split caches...")
            for dataset in self._datasets:
                dataset.setup_cache()
            Console.out("Cache build complete.")

        return valid_cache

    @classmethod
    def _generate_from_config(cls, config: IGConfig, cache: IGConverterCache) -> Path:
        """
        Perform feature extraction and convert the resulting HDF5 file into Parquet format.

        This is called only when no cached data is available.

        Args:
            config (IGConfig): IceGraph configuration object containing user settings.
            cache (IGConverterCache): The cache handler to manage and register the conversion.

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

