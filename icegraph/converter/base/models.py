# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

import os
from abc import ABC, abstractmethod


class Converter(ABC):

    def __init__(self, path: str, **kwargs) -> None:
        """Base class for file conversion."""
        self.path = path

        _dir = self.path if os.path.isdir(self.path) else os.path.dirname(self.path)
        self.outdir = kwargs.get(
            "outdir",
            os.path.join(_dir, self.io_extensions[1])
        )

        os.makedirs(self.outdir, exist_ok=True)

    @abstractmethod
    def convert(self) -> str:
        """Implements the conversion functionality. Returns the path to the output file/dir."""
        pass

    @property
    @abstractmethod
    def io_extensions(self) -> list[str]:
        """Must return a list of string representations of the input and output file extensions,
        i.e.: [<input_file_ext>, <output_file_ext>], omitting the starting period.
        """
        pass
