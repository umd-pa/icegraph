# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from icegraph.exceptions import IceGraphError


class GeometryFrameNotFound(IceGraphError):
    """Error raised when a Geometry frame is not found in the provided GCD file."""
