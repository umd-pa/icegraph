# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from icegraph.exceptions import IceGraphError


class DataError(IceGraphError):
    """Raised when an error occurs during the data loading process."""


class EmptyDatasetError(DataError):
    """Exception raised when a dataset is empty."""


class MissingFieldError(DataError):
    """Exception raised when there is a missing field."""


class PipelineBuildError(IceGraphError):
    """Exception raised when there is an error during pipeline build and configuration."""
