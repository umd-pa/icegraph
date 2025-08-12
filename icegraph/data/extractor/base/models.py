# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, Optional, Sequence, List

from icegraph.config import IGConfig
from icegraph.pathutils import PathValidator, PathResolver
from .exceptions import MissingI3FilesError
from icegraph.console import Console

__all__ = ["IGExtractor"]


class IGExtractor(ABC):
    """
    Abstract base class for data extraction pipelines.
    """

    def __init__(self, source: Union[str, Path, Sequence[Union[str, Path]]]) -> None:
        """
        Initialize the base extractor.

        Args:
            source (Union[str, Path, Sequence[Union[str, Path]]]): Path or sequence of paths to I3 files or a directory containing I3 files.
        """
        self._config: IGConfig = IGConfig.get()

        # save source input
        self._source = source
        self._file_paths: Optional[List[Path]] = None

        # validate input gcd path
        PathValidator.is_valid_file(self._config.gcd_path)

        # Derive output directory next to the input
        resolver = PathResolver(None, origin=None, extension=None, stage="extractor")
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
