# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from icegraph.exceptions import IceGraphError


class MissingI3FilesError(IceGraphError, FileNotFoundError):
    """Raised when no I3 files are found in the input directory."""
