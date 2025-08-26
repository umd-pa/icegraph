# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from abc import abstractmethod, ABC
from typing import Union, Optional
from pathlib import Path

from icegraph.utils.pathutils import PathValidator

__all__ = ["IGMerger"]


class IGMerger(ABC):
    """
    Base class for file merging operations.
    """

    file_ext: Optional[str] = None

    def __init__(self, indir: Union[str, Path]) -> None:
        self.indir = Path(indir)
        PathValidator.is_valid_dir(self.indir)

        # Verify that the subclass defined the required file extension
        if self.file_ext is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} must define the 'file_ext' class attribute."
            )

        self.files: list[Path] = list(self.indir.glob(f"*.{self.file_ext}"))

    @abstractmethod
    def merge(self, outfile: Optional[Union[str, Path]] = None) -> Path:
        ...