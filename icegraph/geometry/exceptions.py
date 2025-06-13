# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean


class GeometryFrameNotFound(Exception):

    def __init__(self, message: str):
        super().__init__(message)
