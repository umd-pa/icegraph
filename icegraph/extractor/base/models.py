# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, Optional

from icegraph.config import Config


class Extractor(ABC):
    """
    Abstract base class for data extraction pipelines..
    """

    def __init__(self, config: Config, input_dir: Optional[Union[str, Path]] = None) -> None:
        """
        Initialize the base extractor.

        Args:
            config (Config): IceGraph configuration object containing user settings.
            input_dir (Optional[Union[str, Path]]): Optional path to override the default input directory.
        """
        self._config: Config = config

        # Use provided input_dir, or fall back to the one in user config
        self.input_dir = Path(input_dir or self._config.user_config.input_dir)

        # Derive output directory next to the input
        base_dir = self.input_dir if self.input_dir.is_dir() else self.input_dir.parent
        self.output_dir = base_dir / "extraction"

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def extract(self) -> Path:
        """
        Run the extraction process and return the output path.

        Returns:
            Path: Path to the output file or directory containing extracted data.

        Raises:
            NotImplementedError: Must be implemented by any subclass of Extractor.
        """
        pass