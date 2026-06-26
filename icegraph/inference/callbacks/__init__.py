# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

# reexport here for convenience
from icegraph.engine.callbacks import CallbackSpec

# base class
from .callback import InferenceCallback

__all__ = [
    "InferenceCallback",
    "CallbackSpec"
]
