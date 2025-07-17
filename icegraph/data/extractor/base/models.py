# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, Optional

from icegraph.config import IGConfig
from icegraph.pathutils import PathValidator, PathResolver

__all__ = ["IGExtractor"]


class IGExtractor(ABC):
    """
    Abstract base class for data extraction pipelines.
    """

    def __init__(self, inpath: Union[str, Path]) -> None:
        """
        Initialize the base extractor.

        Args:
            inpath (Union[str, Path]): Path to the input directory or file.
        """
        self._config: IGConfig = IGConfig.get()

        # Use provided resource
        self.inpath = Path(inpath)

        # validate input paths
        PathValidator.is_valid_path(self.inpath)
        PathValidator.is_valid_file(self._config.gcd_path)

        # Derive output directory next to the input
        resolver = PathResolver(None, origin=inpath, extension=None, stage="extractor")
        self.outdir = resolver.resolve(return_dir=True)

    def __call__(self, outfile: Optional[Union[str, Path]] = None) -> Path:
        return self.extract(outfile)

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
