# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import Self, Type, TYPE_CHECKING, Union
from pathlib import Path

import torch_geometric as pyg

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
        "train_dataset": ("train", TrainingDataset),
        "val_dataset": ("validation", ValidationDataset),
        "test_dataset": ("test", TestDataset),
    }

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
        """
        self._train_dataset = train_dataset
        self._validation_dataset = validation_dataset
        self._test_dataset = test_dataset

        self._datasets = [self._train_dataset, self._validation_dataset, self._test_dataset]

        # get training params from config
        self._config = IGConfig.get()

        self.batch_size = self._config.user_config.training.batch_size
        self.num_workers = self._config.user_config.training.num_workers

        # verify the datasets were passed in the correct order
        if self._train_dataset.subset != "train":
            raise ValueError("Expected train_dataset.subset == 'train'")
        if self._validation_dataset.subset != "validation":
            raise ValueError("Expected val_dataset.subset == 'validation'")
        if self._test_dataset.subset != "test":
            raise ValueError("Expected test_dataset.subset == 'test'")

    def __len__(self) -> int:
        """
        Return the number of events in the full dataset.

        Returns:
            int: Number of events.
        """
        return sum(map(len, self._datasets))

    # --- Static property stubs for type checking and autocompletion ---
    if TYPE_CHECKING:
        @property
        def train_dataset(self) -> TrainingDataset: ...

        @property
        def val_dataset(self) -> ValidationDataset: ...

        @property
        def test_dataset(self) -> TestDataset: ...

    @property
    def train_dataloader(self) -> pyg.loader.DataLoader:
        """
        Returns a Torch Geometric dataloader for the training split.
        """
        return self.train_dataset.dataloader(
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    @property
    def val_dataloader(self) -> pyg.loader.DataLoader:
        """
        Returns a Torch Geometric dataloader for the validation split.
        """
        return self.val_dataset.dataloader(
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    @property
    def test_dataloader(self) -> pyg.loader.DataLoader:
        """
        Returns a Torch Geometric dataloader for the test split.
        """
        return self.test_dataset.dataloader(
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    @classmethod
    def load_from_lmdb(cls, train_lmdb: Union[str, Path], val_lmdb: Union[str, Path], test_lmdb: Union[str, Path]) -> Self:
        """
        Load datasets from LMDB files and create an instance of the dataset registry.

        Args:
            train_lmdb (Union[str, Path]): Path to the LMDB file containing the training dataset.
            val_lmdb (Union[str, Path]): Path to the LMDB file containing the validation dataset.
            test_lmdb (Union[str, Path]): Path to the LMDB file containing the test dataset.

        Returns:
            Self: An instance of the class initialized with training, validation, and test datasets.
        """
        return cls(TrainingDataset(train_lmdb), ValidationDataset(val_lmdb), TestDataset(test_lmdb))

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

