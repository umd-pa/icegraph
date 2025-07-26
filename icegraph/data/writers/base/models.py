# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from abc import abstractmethod, ABC
from typing import Union, Optional
from pathlib import Path
import uuid

import pandas as pd

__all__ = ["IGWriter"]


class IGWriter(ABC):
    """
    Abstract base class for writing IceGraph data tables to various formats.

    Subclasses must implement the `write` method, which handles serialization of
    the internal DataFrame (`self.table`) to a specific output format (e.g., LMDB, Parquet).
    """

    def __init__(self, outfile: Optional[Union[str, Path]], mode: str = "w"):
        """
        Initialize the writer with a DataFrame to serialize.

        Args:
            outfile (Optional[Union[str, Path]]): The destination path for the output file.
            mode (str): Write mode to use; "w" for write and "a" for append.
        """
        self.outfile = outfile
        self.mode = mode

        # create a temp file to write atomically
        self.tmp_path = self.outfile.parent / f".{self.outfile.name}.{uuid.uuid4().hex}.tmp"
        if self.tmp_path.exists():
            self.tmp_path.unlink()

        # don't delete any files if appending
        if mode == "w" and self.outfile.exists():
            self.outfile.unlink()

    @abstractmethod
    def write(self, table: pd.DataFrame, include_cols: Optional[list] = None):
        """
        Abstract method to write the data to a file.

        Args:
            table (pd.DataFrame): A pandas DataFrame object to write.
            include_cols (Optional[list], optional): A list of column names to include
                in the written output. If None, all columns should be written.

        Raises:
            NotImplementedError: If the method is not overridden in a subclass.
        """
        ...