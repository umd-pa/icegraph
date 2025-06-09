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

    def __init__(self, config: IGConfig, input_file: Union[str, Path], output_dir: Optional[Union[str, Path]] = None) -> None:
        """
        Initialize the base converter for file transformation tasks.

        Args:
            config (IGConfig): IceGraph configuration object containing user settings.
            input_file (Union[str, Path]): Path to the input file or directory.
            output_dir (Optional[Union[str, Path]]): Optional custom root directory for output files.

        Raises:
            NotImplementedError: If the subclass has not defined `out_extension`.
        """
        self._config: IGConfig = config
        self.input_file = Path(input_file)

        # Determine base directory from input path
        base_dir = self.input_file if self.input_file.is_dir() else self.input_file.parent

        # Verify that the subclass defined the required output extension
        if self.out_extension is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} must define the 'out_extension' class attribute."
            )

        # Generate a unique identifier for this input, typically based on its content
        input_hash = self._config.get_input_state_hash()

        # Determine the full output path based on the hash and chosen extension
        output_root = Path(output_dir or base_dir / self.out_extension)
        self.outdir = output_root / input_hash

        # Create output directory if it doesn't exist
        self.outdir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def convert(self) -> Path:
        """
        Run the conversion process defined by the subclass.

        Returns:
            Path: Path to the converted output file or directory.

        Raises:
            NotImplementedError: Must be implemented by any subclass of Converter.
        """
        ...
