# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from icegraph.exceptions import IceGraphError


class ProcessorError(IceGraphError):
    """Raised when there is an exception during processing."""


class VectorMappingError(ProcessorError):
    """Raised when an exception is raised during vector mapping generation."""
