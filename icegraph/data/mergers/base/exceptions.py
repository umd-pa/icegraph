# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from icegraph.exceptions import IceGraphError


class MergeError(IceGraphError):
    """Raised when an error occurs during the merging process."""


class MergeToolNotFoundError(IceGraphError):
    """Raised when a merge tool cannot be found."""


class MissingHDF5FilesError(IceGraphError, FileNotFoundError):
    """Raised when no HDF5 files are found in the input directory."""


class MissingLMDBFilesError(IceGraphError, FileNotFoundError):
    """Raised when no LMDB files are found in the input directory."""
