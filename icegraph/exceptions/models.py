# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

__all__ = ["IceGraphError"]


# project base exception
class IceGraphError(Exception):
    """Base exception for IceGraph."""

    def __init__(self, message: str):
        super().__init__(message)
