# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from . import (
    console,
    data,
    config,
    geometry,
    renderer,
    trainer,
    utils,
    exceptions,
    inference
)

__all__ = [
    "console",
    "data",
    "config",
    "renderer",
    "trainer",
    "geometry",
    "utils",
    "exceptions",
    "inference"
]

from ._version import __version__

# set up a null handler to prevent errors on library import where
# no logger has been configured
import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())

