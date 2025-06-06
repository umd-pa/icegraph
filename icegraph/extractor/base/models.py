# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from abc import ABC, abstractmethod
import os

from icegraph import config


class Extractor(ABC):

    def __init__(self, path: str, **kwargs) -> None:
        self.path = path

        _dir = self.path if os.path.isdir(self.path) else os.path.dirname(self.path)
        self.outdir = kwargs.get(
            "outdir",
            os.path.join(_dir, config.EXTRACTOR_OUTDIR_NAME)
        )

        os.makedirs(self.outdir, exist_ok=True)

    @abstractmethod
    def extract(self) -> str:
        """Implements the data extraction functionality. Returns the path to the output file/dir."""
        pass
