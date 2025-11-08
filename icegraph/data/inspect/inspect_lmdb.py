# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import Union
from pathlib import Path

from icegraph.data.readers import LMDBReader


class InspectLMDB:
    """Small utility for inspecting LMDB files."""

    def __init__(self, infile: Union[str, Path]) -> None:
        with LMDBReader(infile) as reader:
            self.df = reader.to_pandas()
            self.attrs = reader.attrs()
