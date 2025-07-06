# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import cast, Optional, Union
from pathlib import Path

import pandas as pd
from scipy.spatial.ckdtree import cKDTree

from icegraph.console import Console
from icegraph.config import IGConfig
from icegraph.console.streams import suppress_stderr
from .schemas import generate_vector_mapping
from icegraph.geometry import Detector
from icegraph.data.writers import LMDBWriter

__all__ = ["GraphSamplePreprocessor"]


class GraphSamplePreprocessor:
    """
    Processes an HDF5 file generated via `ml_suite`, saves graph samples to a Lightning Memory-Mapped Database (LMDB) file.
    """

    def __init__(self, input_file: Union[str, Path]) -> None:
        """
        Initialize the data processor with an input file.

        Args:
            input_file (Union[str, Path]): Path to the input file or directory.

        """
        self.input_file = Path(input_file)
        self._config: IGConfig = IGConfig.get()

    def __call__(self) -> Path:
        """
        Invoke the processor and return the processed output.

        Returns:
            Path: Path to the output LMDB file.
        """
        return self.process()

    def process(self, outfile: Optional[Union[str | Path]] = None) -> Path:
        """
        Runs the sample building script. Saves the resulting samples to an LMDB file.

        Returns:
            Path: Path to the output LMDB file.
        """
        Console.out(f"Building graph samples from raw data: {self.input_file}")
        Console.spinner().start()

        outfile = Path(outfile or self.input_file.parent / "graphs.lmdb")

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
        vector_map = generate_vector_mapping()
        self._apply_column_map(features_table, vector_map)

        # compress features by event
        features_table = self._compress(features_table)

        # merge the truth and features tables and calculate edge indices
        table = self._merge_tables(features_table, truth_table)
        table = self._append_edge_index(table)

        Console.spinner().stop()
        Console.out(f"Build complete, saving to LMDB...")

        # export to lmdb
        self._to_lmdb(table, outfile)
        Console.out(f"Output file saved to {outfile}")

        return outfile

    def _append_edge_index(self, table: pd.DataFrame) -> pd.DataFrame:
        """
        Compute k-nearest neighbor graph edges and distances for each event's DOM features.

        For each row (i.e., event) in the table, this method constructs an edge index and
        corresponding edge weights using k-nearest neighbor (k=10) on DOM spatial coordinates.
        The resulting graph structure is appended as two new columns: 'edge_index' and 'edge_weight'.

        Args:
            table (pd.DataFrame): Input table with a 'features' column, where each entry is a list
                                  of per-DOM feature dictionaries.

        Returns:
            pd.DataFrame: The input table with two new columns:
                          - 'edge_index': List of [2, num_edges] indices.
                          - 'edge_weight': List of distances between connected DOMs.
        """
        # Precompute key order from the first row
        sample_dict = table.iloc[0]['features']
        key_order = list(sample_dict[0].keys())

        dom_pos_cols = self._config.standard_id_col_config.dom_position_columns

        def compute_edges(row: pd.Series) -> pd.Series:
            features_dict = row["features"]

            # extract dom positions
            dom_positions = [[f[c] for c in dom_pos_cols] for f in features_dict]

            # define the kdtree
            tree = cKDTree(dom_positions)

            # KNN (k+1 to skip self)
            _, neighbors = tree.query(dom_positions, k=11)

            # Build edge list and weights
            edge_index = []
            edge_weight = []

            for i, nbrs in enumerate(neighbors):
                for j in nbrs[1:]:  # skip self (first neighbor)
                    edge_index.append([i, j])
                    dist = sum((a - b) ** 2 for a, b in zip(dom_positions[i], dom_positions[j])) ** 0.5
                    edge_weight.append(dist)

            # Transpose edge_index to [2, num_edges] format as list-of-lists
            edge_index = list(map(list, zip(*edge_index)))

            return pd.Series({
                "edge_index": edge_index,
                "edge_weight": edge_weight
            })

        # Apply to all rows
        table[['edge_index', 'edge_weight']] = table.apply(compute_edges, axis=1)
        return table

    def _merge_tables(self, features: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
        """
        Merge the feature and truth tables on the event ID columns.

        This method performs a left join on the feature table using the configured event ID
        columns, appending the corresponding truth values to each event.

        Args:
            features (pd.DataFrame): The DataFrame containing per-event feature lists.
            truth (pd.DataFrame): The DataFrame containing truth labels per event.

        Returns:
            pd.DataFrame: The merged DataFrame with both features and corresponding labels.

        Raises:
            AssertionError: If the number of rows in `features` and `truth` does not match,
                            indicating a potential ID mismatch.
        """
        assert len(features) == len(truth), "ID collision, please rerun feature extraction."

        table = pd.merge(
            features,
            truth,
            on=self._config.standard_id_col_config.event_id_columns,
            how='left'
        )

        return table

    def _reshape_features_table(self, table: pd.DataFrame) -> pd.DataFrame:
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
        table = table.reset_index()

        # Create graphs
        dom_id_cols = self._config.standard_id_col_config.dom_id_columns
        dom_pos_cols = self._config.standard_id_col_config.dom_position_columns

        detector = Detector()

        # apply & expand into separate columns
        coords_df = table.apply(
            lambda row: pd.Series(
                detector.get_dom_coords(
                    row[dom_id_cols[0]],
                    row[dom_id_cols[1]],
                    row[dom_id_cols[2]]
                ),
                index=dom_pos_cols
            ),
            axis=1
        )

        table = pd.concat([table, coords_df], axis=1, ignore_index=False)
        table.drop(columns=dom_id_cols, inplace=True)

        return table

    def _compress(self, table: pd.DataFrame) -> pd.DataFrame:
        """
        Group DOM-level features into a list for each event and return an event-level table.

        This method converts rows corresponding to individual DOMs into a single list of
        DOM feature dictionaries per event. The resulting table has one row per event.

        Args:
            table (pd.DataFrame): The DOM-level feature table.

        Returns:
            pd.DataFrame: The event-level feature table with one list of features per event.
        """
        dom_id_cols = self._config.standard_id_col_config.dom_id_columns
        event_id_cols = self._config.standard_id_col_config.event_id_columns
        dom_pos_cols = self._config.standard_id_col_config.dom_position_columns

        feature_cols = [c for c in table.columns if c not in dom_id_cols + event_id_cols]

        table['features'] = table[feature_cols + dom_pos_cols].to_dict(orient='records')

        table = (
            table
            .groupby(event_id_cols, as_index=False)
            .agg(features=('features', list))
        )

        return table

    def _to_lmdb(self, table: pd.DataFrame, outfile: Union[str, Path]) -> None:
        """
        Writes the given DataFrame to an LMDB file.

        Args:
            table (pd.DataFrame): Data to write.
            outfile (str): Output file path. Defaults to "graphs.lmdb" in the source dir.
        """
        dom_id_cols = self._config.standard_id_col_config.dom_id_columns
        event_id_cols = self._config.standard_id_col_config.event_id_columns

        include_cols = [c for c in table.columns if c not in dom_id_cols + event_id_cols]

        # write to lmdb
        writer = LMDBWriter(table)
        writer.write(outfile, include_cols)

    @staticmethod
    def _apply_column_map(table: pd.DataFrame, mapping: dict) -> None:
        """
        Renames the columns of a DataFrame using the provided mapping.

        Args:
            table (pd.DataFrame): DataFrame to modify.
            mapping (dict): Mapping from original column names to new names.
        """
        table.rename(columns=mapping, inplace=True)