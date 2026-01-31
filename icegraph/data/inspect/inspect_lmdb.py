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

    def to_pandas(self) -> pd.DataFrame:
        """Load all data records from the LMDB into a pandas DataFrame."""
        records: list[dict[str, ArrayG]] = []
        with self.env.begin(db=self.dbs["data"]) as txn, txn.cursor() as cursor:
            # deserialize the data
            for _, data_packed in cursor:
                row = self._unpack(data_packed)
                records.append(row)

        # build a DataFrame from records
        return pd.DataFrame.from_records(records) if records else pd.DataFrame()
