# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from abc import abstractmethod, ABC
from typing import Union, Optional, Dict
from pathlib import Path
import uuid

import pandas as pd

__all__ = ["Writer"]


class Writer(ABC):
    """
    Abstract base class for writing IceGraph data tables to various formats.

    Subclasses must implement the `write` method, which handles serialization of
    the internal DataFrame (`self.table`) to a specific output format (e.g., LMDB, Parquet).
    """

    suffix: Optional[str] = None

    def __init__(self, outfile: Optional[Union[str, Path]]):
        """
        Initialize the writer with a DataFrame to serialize.

        Args:
            outfile (Optional[Union[str, Path]]): The destination path for the output file.
        """
        self.outfile = outfile

        # create a temp file to write atomically
        self.tmp_path = self.outfile.parent / f".{self.outfile.name}.{uuid.uuid4().hex}.tmp"
        if self.tmp_path.exists():
            self.tmp_path.unlink()

        if self.outfile.exists():
            self.outfile.unlink()

    def __init_subclass__(cls, **kwargs):
        if cls.suffix is None:
            raise NotImplementedError("All subclasses of Writer must implement the class attribute 'suffix'.")

    @abstractmethod
    def write(self, df: pd.DataFrame):
        """
        Abstract method to write the data to a file.

        Args:
            df (pd.DataFrame): A pandas DataFrame object to write.

        Raises:
            NotImplementedError: If the method is not overridden in a subclass.
        """
        ...

    @abstractmethod
    def write_attrs(self, groups: Dict[str, Dict[str, str]], include_defaults: bool = True) -> None:
        ...