# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from abc import abstractmethod, ABC
from typing import Union, Optional
from pathlib import Path


class IGMerger(ABC):
    """
    Base class for file merging operations.
    """

    file_ext: Optional[str] = None

    def __init__(self, input_dir: Union[str, Path]) -> None:
        self.input_dir = Path(input_dir)

        # Verify that the subclass defined the required file extension
        if self.file_ext is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} must define the 'file_ext' class attribute."
            )

        self.output_file = self.input_dir / f"merged.{self.file_ext}"
        self.files = self.input_dir.glob(f"*.{self.file_ext}")

        # verify output directory exists
        self.output_file.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def merge(self) -> Path:
        ...