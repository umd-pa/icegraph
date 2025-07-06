# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, Optional

from icegraph.config import IGConfig


__all__ = ["IGConverter"]

class IGConverter(ABC):
    out_extension: str = None
    """File extension or format identifier used by subclasses to define output type (e.g., 'hdf5', 'parquet')."""

    def __init__(self, input_file: Union[str, Path]) -> None:
        """
        Initialize the base converter for file transformation tasks.

        Args:
            input_file (Union[str, Path]): Path to the input file or directory.

        Raises:
            NotImplementedError: If the subclass has not defined `out_extension`.
        """
        self.input_file = Path(input_file)
        self._config: IGConfig = IGConfig.get()

        # Verify that the subclass defined the required output extension
        if self.out_extension is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} must define the 'out_extension' class attribute."
            )

    def __call__(self):
        return self.convert()

    @abstractmethod
    def convert(self, outfile: Optional[Union[str | Path]] = None) -> Path:
        """
        Run the conversion process defined by the subclass.

        Returns:
            Path: Path to the converted output file or directory.

        Raises:
            NotImplementedError: Must be implemented by any subclass of Converter.
        """
        ...
