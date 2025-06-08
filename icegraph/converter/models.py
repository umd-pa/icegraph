# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

import pandas as pd
from typing import cast
from pathlib import Path

from icegraph.console import Console
from icegraph.console.streams import suppress_stderr
from .schemas import MLSuiteVectorMapping
from .base import Converter


class HDF5ToParquet(Converter):
    """
    Converts an HDF5 file generated via `ml_suite` into Parquet format.

    The input file is assumed to contain 'features' and 'truth' tables, which are
    saved as separate Parquet files in the output directory.
    """

    out_extension = "parquet"

    def convert(self) -> Path:
        """
        Converts an HDF5 input file to Parquet format.

        Returns:
            Path: Path to the output directory containing converted Parquet files.
        """
        Console.out(f"Converting to {self.out_extension}: {self.input_file}")
        Console.spinner().start()

        # Load data to DataFrames
        # IDE might complain these aren't DataFrames; they are.
        # Suppressing very loud HDF5 mismatched header warning
        with suppress_stderr():
            features_table = cast(pd.DataFrame, pd.read_hdf(
                self.input_file,
                key=self._config.user_config.table_names.features
            ))
            truth_table = cast(pd.DataFrame, pd.read_hdf(
                self.input_file,
                key=self._config.user_config.table_names.truth
            ))

        # Run reshaping
        features_table = self._reshape_features_table(features_table)
        truth_table = self._reshape_truth_table(truth_table)

        # Apply feature vector mapping
        vector_map = MLSuiteVectorMapping(self._config)
        self._apply_column_map(features_table, vector_map.get_mapping())

        # Export to Parquet
        self._to_parquet(features_table, "features")
        self._to_parquet(truth_table, "truth")

        Console.spinner().stop()
        Console.out(f"Output files saved to {self.outdir}")

        return self.outdir

    def _reshape_features_table(self, table: pd.DataFrame) -> pd.DataFrame:
        """
        Reshapes the features table by generating composite keys and pivoting
        ml_suite generated vector data.

        Args:
            table (pd.DataFrame): Input features table.

        Returns:
            pd.DataFrame: Reshaped features table.
        """
        id_columns = ['Run', 'Event', 'SubEvent', 'SubEventStream', 'exists', 'string', 'om']
        table = self._replace_with_composite_keys(table, id_columns, "id")
        table.drop(columns=["pmt"], inplace=True)

        # Move id to first column for readability
        table.insert(0, 'id', table.pop('id'))

        # Pivot the table
        table = table.pivot_table(index='id', columns='vector_index', values='item', aggfunc="first")
        return table

    def _reshape_truth_table(self, table: pd.DataFrame) -> pd.DataFrame:
        """
        Reshapes the truth table by generating composite keys.

        Args:
            table (pd.DataFrame): Input truth table.

        Returns:
            pd.DataFrame: Reshaped truth table.
        """
        id_columns = ['Run', 'Event', 'SubEvent', 'SubEventStream', 'exists']
        table = self._replace_with_composite_keys(table, id_columns, "id")
        table.insert(0, 'id', table.pop('id'))
        return table

    @staticmethod
    def _replace_with_composite_keys(table: pd.DataFrame, id_columns: list[str], new_column_name: str) -> pd.DataFrame:
        """
        Replaces multiple identifier columns with a single composite key.

        Args:
            table (pd.DataFrame): Input table with identifier columns.
            id_columns (list[str]): List of columns to combine.
            new_column_name (str): Name for the new composite column.

        Returns:
            pd.DataFrame: Modified table with composite key.
        """
        table[new_column_name] = table[id_columns].astype(str).agg(
            lambda row: '|'.join(f'{col}={val}' for col, val in zip(id_columns, row)), axis=1
        )
        table.drop(columns=id_columns, inplace=True)
        return table

    def _to_parquet(self, table: pd.DataFrame, name: str) -> None:
        """
        Writes the given DataFrame to a Parquet file in the output directory.

        Args:
            table (pd.DataFrame): Data to write.
            name (str): Output file name (e.g., 'features', 'truth').
        """
        output_path = self.outdir / f"{name}.{self.out_extension}"
        table.to_parquet(output_path)

    @staticmethod
    def _apply_column_map(table: pd.DataFrame, mapping: dict) -> None:
        """
        Renames the columns of a DataFrame using the provided mapping.

        Args:
            table (pd.DataFrame): DataFrame to modify.
            mapping (dict): Mapping from original column names to new names.
        """
        table.rename(columns=mapping, inplace=True)