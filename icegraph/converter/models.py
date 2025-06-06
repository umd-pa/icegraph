# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

import os
import pandas as pd

from icegraph.console import Console
from icegraph.console.streams import suppress_stderr
from icegraph import config
from .schemas import MLSuiteVectorMapping
from .base import Converter


class HDF5ToParquet(Converter):
    """Takes in one hdf5 file (generated via ml_suite) and converts it to parquet format."""

    @property
    def io_extensions(self):
        return ["hdf5", "parquet"]

    def convert(self):
        Console.out(f"Converting file ({self.path}) to {self.io_extensions[1]}")
        Console.spinner().start()

        # load data to dataframes
        # IDE might complain these aren't dataframes; they are
        # suppressing very loud HDF5 mismatched header warning
        with suppress_stderr():
            features_table: pd.DataFrame = pd.read_hdf(self.path, key=config.FEATURES_TABLE_NAME)
            truth_table: pd.DataFrame = pd.read_hdf(self.path, key=config.TRUTH_TABLE_NAME)

        # run reshaping
        features_table = self._reshape_features_table(features_table)
        truth_table = self._reshape_truth_table(truth_table)

        vector_map = MLSuiteVectorMapping(
            os.path.join(config.CONFIG_DIR, "extraction/feature_extraction.yaml"),
            os.path.join(config.CONFIG_DIR, "extraction/features_map.yaml")
        )

        self._apply_column_map(features_table, vector_map.get_mapping())

        self._to_parquet(features_table, "features")
        self._to_parquet(truth_table, "truth")

        Console.spinner().stop()
        Console.out(f"Output files saved to {self.outdir}")

        return self.outdir

    def _reshape_features_table(self, table: pd.DataFrame) -> pd.DataFrame:
        # generate composite keys
        id_columns = ['Run', 'Event', 'SubEvent', 'SubEventStream', 'exists', 'string', 'om']
        table = self._replace_with_composite_keys(table, id_columns, "id")
        table.drop(columns=["pmt"])

        # move id to first column for readability
        table.insert(0, 'id', table.pop('id'))

        # pivot the table
        table = table.pivot_table(index='id', columns='vector_index', values='item', aggfunc="first")
        return table

    def _reshape_truth_table(self, table: pd.DataFrame) -> pd.DataFrame:
        # generate composite keys
        id_columns = ['Run', 'Event', 'SubEvent', 'SubEventStream', 'exists']
        table = self._replace_with_composite_keys(table, id_columns, "id")

        # move id to first column for readability
        table.insert(0, 'id', table.pop('id'))
        return table

    @staticmethod
    def _replace_with_composite_keys(table: pd.DataFrame, id_columns: list[str], new_column_name: str) -> pd.DataFrame:
        """Generates a composite key for each row."""
        table[new_column_name] = table[id_columns].astype(str).agg(
            lambda row: '|'.join(f'{col}={val}' for col, val in zip(id_columns, row)), axis=1
        )
        table = table.drop(columns=id_columns)
        return table

    def _to_parquet(self, table: pd.DataFrame, name: str) -> None:
        table.to_parquet(os.path.join(self.outdir, f"{name}.{self.io_extensions[1]}"))

    @staticmethod
    def _apply_column_map(table: pd.DataFrame, mapping: dict) -> None:
        """Rename columns using a provided dictionary mapping."""
        table.rename(columns=mapping, inplace=True)
