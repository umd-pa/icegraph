# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from icegraph.exceptions import IceGraphError


class TrainerError(IceGraphError):
    """Raised when an exception occurs during training."""


class UnknownModelError(TrainerError):
    """Raised when the user attempts to use a model that is not registered in the ModelFactory."""


class EmptyDataLoaderError(TrainerError):
    """Raised when the trainer encounters a PyG DataLoader with no data."""

