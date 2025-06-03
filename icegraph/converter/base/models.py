# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

import os
from abc import ABC, abstractmethod


class Converter(ABC):

    def __init__(self, path: str, **kwargs) -> None:
        """Base class for file conversion."""
        self.path = path
        self.outdir = kwargs.get(
            "outdir",
            os.path.join(os.path.dirname(self.path), self.to_filetype)
        )

        os.makedirs(self.outdir, exist_ok=True)

    @abstractmethod
    def convert(self) -> None:
        """Implements the conversion functionality."""
        pass

    @property
    @abstractmethod
    def to_filetype(self) -> str:
        """Must return a string representation of the file extension of the destination file type."""
        pass
