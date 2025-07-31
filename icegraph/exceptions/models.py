# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

__all__ = ["IceGraphError", "IceCubeImportError"]


# project base exception
class IceGraphError(Exception):
    """Base exception for IceGraph."""

    def __init__(self, message: str):
        super().__init__(message)


class IceCubeImportError:

    class IceCubeMissingBase:
        def __init__(self, *args, **kwargs):
            raise ImportError("IceCube base class is missing. Please activate the IceTray environment.")

    def __getattr__(self, name):
        raise ImportError(
            f"Cannot access '{name}', IceCube dependencies are missing or the IceTray environment is not active."
        )

    def __call__(self, *args, **kwargs):
        raise ImportError("Attempted to call a missing IceCube object. Please activate the IceTray environment.")

    def __getitem__(self, key):
        raise ImportError("Attempted to index a missing IceCube object. Please activate the IceTray environment.")
