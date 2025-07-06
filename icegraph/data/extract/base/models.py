# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, Optional

from icegraph.config import IGConfig

__all__ = ["IGExtractor"]


class IGExtractor(ABC):
    """
    Abstract base class for data extraction pipelines.
    """

    def __init__(self, input_dir: Union[str, Path]) -> None:
        """
        Initialize the base extractor.

        Args:
            input_dir (Union[str, Path]): Path to the input directory.
        """
        self._config: IGConfig = IGConfig.get()

        # Use provided input_dir
        self.input_dir = Path(input_dir)

        # Derive output directory next to the input
        base_dir = self.input_dir if self.input_dir.is_dir() else self.input_dir.parent
        self.output_dir = base_dir / "extraction"

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def __call__(self):
        return self.extract()

    @abstractmethod
    def extract(self, outfile: Optional[Union[str, Path]] = None) -> Path:
        """
        Run the extraction process and return the output path.

        Returns:
            Path: Path to the output file or directory containing extracted data.

        Raises:
            NotImplementedError: Must be implemented by any subclass of Extractor.
        """
        ...
