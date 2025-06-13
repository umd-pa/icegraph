# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from .base import IGData


class TrainingDataset(IGData):
    """
    Dataset class for the training split.

    Inherits from IGData and sets the required class attribute `subset = "train"`,
    which triggers filtering of the truth table to only include training events.
    """
    subset = "train"


class ValidationDataset(IGData):
    """
    Dataset class for the validation split.

    Inherits from IGData and sets the required class attribute `subset = "validation"`,
    which triggers filtering of the truth table to only include validation events.
    """
    subset = "validation"


class TestDataset(IGData):
    """
    Dataset class for the test split.

    Inherits from IGData and sets the required class attribute `subset = "test"`,
    which triggers filtering of the truth table to only include test events.
    """
    subset = "test"