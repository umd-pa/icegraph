# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from .base import IGData

__all__ = ["TrainingDataset", "ValidationDataset", "TestDataset"]


class TrainingDataset(IGData):
    """
    Dataset class for the training split.
    """
    subset = "train"


class ValidationDataset(IGData):
    """
    Dataset class for the validation split.
    """
    subset = "validation"


class TestDataset(IGData):
    """
    Dataset class for the test split.
    """
    subset = "test"