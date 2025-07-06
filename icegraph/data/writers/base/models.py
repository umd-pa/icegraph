# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from abc import abstractmethod, ABC
from typing import Union, Optional
from pathlib import Path

import pandas as pd

__all__ = ["IGWriter"]


class IGWriter(ABC):
    """
    Abstract base class for writing IceGraph data tables to various formats.

    Subclasses must implement the `write` method, which handles serialization of
    the internal DataFrame (`self.table`) to a specific output format (e.g., LMDB, Parquet).
    """

    def __init__(self, table: pd.DataFrame):
        """
        Initialize the writer with a DataFrame to serialize.

        Args:
            table (pd.DataFrame): The DataFrame containing the graph-structured data
                to be written out to disk.
        """
        self.table = table

    @abstractmethod
    def write(self, outfile: Union[str, Path], include_cols: Optional[list] = None):
        """
        Abstract method to write the data to a file.

        Args:
            outfile (Union[str, Path]): The destination path for the output file.
            include_cols (Optional[list], optional): A list of column names to include
                in the written output. If None, all columns should be written.

        Raises:
            NotImplementedError: If the method is not overridden in a subclass.
        """
        ...