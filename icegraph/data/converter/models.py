# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import cast
from pathlib import Path

import pandas as pd

from icegraph.console import Console
from icegraph.console.streams import suppress_stderr
from .schemas import generate_vector_mapping
from .base import IGConverter

__all__ = ["HDF5ToParquet"]


class HDF5ToParquet(IGConverter):
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

        # Apply feature vector mapping
        vector_map = generate_vector_mapping(self._config)
        self._apply_column_map(features_table, vector_map)

        event_id_cols = self._config.standard_id_col_config.event_id_columns

        features_table.sort_values(event_id_cols, inplace=True, kind="mergesort")
        truth_table.sort_values(event_id_cols, inplace=True, kind="mergesort")

        # Export to Parquet
        self._to_parquet(features_table.reset_index(), "features")
        self._to_parquet(truth_table.reset_index(), "truth")

        Console.spinner().stop()
        Console.out(f"Output files saved to {self.outdir}")

        return self.outdir

    @staticmethod
    def _reshape_features_table(table: pd.DataFrame) -> pd.DataFrame:
        """
        Reshapes the features table by pivoting ml_suite generated vector data.

        Args:
            table (pd.DataFrame): Input features table.

        Returns:
            pd.DataFrame: Reshaped features table.
        """
        # Pivot the table
        pivot_col = "vector_index"
        value_col = "item"
        index_cols = [c for c in table.columns if c not in {pivot_col, value_col}]

        table = table.pivot_table(index=index_cols, columns=pivot_col, values=value_col, aggfunc="first")

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