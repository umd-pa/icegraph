# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from icegraph.exceptions import IceGraphError


class WriterError(IceGraphError):
    """Exception raised when an error occurs on write."""
