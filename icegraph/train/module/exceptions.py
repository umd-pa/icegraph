# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean


class InvalidModel(Exception):
    """Exception called when an invalid model name is passed."""

    def __init__(self, message):
        super().__init__(message)