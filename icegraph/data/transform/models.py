# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import cast, Optional, Union
from pathlib import Path
import json

import pandas as pd
import numpy as np
import torch
from torch_cluster import knn_graph
from scipy.spatial.ckdtree import cKDTree
from sklearn.preprocessing import StandardScaler

from icegraph.console import Console
from icegraph.config import IGConfig
from icegraph.console.streams import suppress_stderr
from .schemas import generate_vector_mapping
from icegraph.geometry import Detector
from icegraph.data.writers import LMDBWriter

__all__ = ["TransformToDataset"]


class TransformToDataset:
    """
    Transforms an HDF5 file of DOM-level event data into a Lightning Memory-Mapped Database (LMDB)
    of compressed graph samples, with edge indices and features for GNN processing.

    This includes:
    - Reshaping DOM-level features
    - Mapping feature vectors
    - Grouping by event
    - Computing graph edge structures
    - Writing to LMDB for training/evaluation
    """

    def __init__(self, infile: Union[str, Path]) -> None:
        """
        Initialize the data processor with an input file.

        Args:
            infile (Union[str, Path]): Path to the input file or directory.

        """
        self.infile = Path(infile)
        self._config: IGConfig = IGConfig.get()

        # get relevant columns from config
        self.dom_id_cols = self._config.standard_id_col_config.dom_id_columns
        self.event_id_cols = self._config.standard_id_col_config.event_id_columns
        self.dom_pos_cols = self._config.standard_id_col_config.dom_position_columns

        # define a simple getter to extract data columns in a table and ignore IDs
        self.data_cols = lambda table: [c for c in table.columns if c not in self.dom_id_cols + self.event_id_cols]

        # get columns we should be applying a log scale to (i.e. energy)
        self.apply_log_scaling = self._config.user_config.data.normalization.apply_log_scaling

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
        Console.banner("Data Transformer")
        Console.out(f"Building graph samples from raw data: {self.infile}")
        Console.spinner().start()

        outfile = Path(outfile or self.infile.parent / "graphs.lmdb")
        outfile.parent.mkdir(parents=True, exist_ok=True)

        # Load data to DataFrames
        # IDE might complain these aren't DataFrames; they are.
        # Suppressing very loud HDF5 mismatched header warning
        with suppress_stderr():
            features_table = cast(pd.DataFrame, pd.read_hdf(
                self.infile,
                key=self._config.user_config.table_names.features
            ))
            truth_table = cast(pd.DataFrame, pd.read_hdf(
                self.infile,
                key=self._config.user_config.table_names.truth
            ))

        # Run reshaping
        features_table = self._reshape_features_table(features_table)

        # Apply feature vector mapping
        vector_map = generate_vector_mapping()
        self._apply_column_map(features_table, vector_map)

        # normalize the data, and save the norms for inversion later
        scaler_params_outfile = outfile.parent / "scaler-params-features.json"
        features_table = self._normalize(features_table, scaler_params_outfile)

        Console.out(
            f"Scaler parameters saved to {scaler_params_outfile}. "
            f"Use this to normalize/denormalize features in production.",
            control_prefix="\r"
        )

        scaler_params_outfile = outfile.parent / "scaler-params-labels.json"
        truth_table = self._normalize(truth_table, scaler_params_outfile)

        Console.out(
            f"Scaler parameters saved to {scaler_params_outfile}. "
            f"Use this to normalize/denormalize labels in production.",
            control_prefix="\r"
        )

        # compress features by event
        features_table = self._compress(features_table)

        # merge the truth and features tables and calculate edge indices
        table = self._merge_tables(features_table, truth_table)
        table = self._append_edge_index(table)

        Console.spinner().stop()
        Console.out("Build complete, writing to LMDB...")

        # export to lmdb
        self._to_lmdb(table, outfile)

        return outfile

    def _append_edge_index(self, table: pd.DataFrame) -> pd.DataFrame:
        """
        Compute k-nearest neighbor graph edges and distances for each event's DOM features
        using PyTorch's `knn_graph`.

        For each row (i.e., event) in the table, this method constructs an edge index and
        corresponding edge weights using k-nearest neighbors (k=10) on DOM spatial coordinates.
        The resulting graph structure is appended as two new columns: 'edge_index' and 'edge_weight'.

        Args:
            table (pd.DataFrame): Input table with a 'features' column, where each entry is a list
                                  of per-DOM feature dictionaries.

        Returns:
            pd.DataFrame: The input table with two new columns:
                          - 'edge_index': List of [2, num_edges] indices.
                          - 'edge_weight': List of distances between connected DOMs.
        """

        def compute_edges(row: pd.Series) -> pd.Series:
            features_dict = row["features"]

            # Extract DOM positions as tensor [num_nodes, num_coords]
            dom_positions = torch.tensor(
                [[f[c] for c in self.dom_pos_cols] for f in features_dict],
                dtype=torch.float
            )

            # Compute kNN graph (k=10 neighbors)
            edge_index = knn_graph(dom_positions, k=10, loop=False)  # shape: [2, num_edges]

            # Compute edge weights (Euclidean distance)
            src, dst = edge_index
            distances = torch.norm(dom_positions[src] - dom_positions[dst], dim=1)  # [num_edges]

            return pd.Series({
                "edge_index": edge_index.tolist(),
                "edge_weight": distances.tolist()
            })

        # Apply to all rows
        table[["edge_index", "edge_weight"]] = table.apply(compute_edges, axis=1)
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
            on=self.event_id_cols,
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

        detector = Detector()

        # apply & expand into separate columns
        coords_df = table.apply(
            lambda row: pd.Series(
                detector.get_dom_coords(
                    row[self.dom_id_cols[0]],
                    row[self.dom_id_cols[1]],
                    row[self.dom_id_cols[2]]
                ),
                index=self.dom_pos_cols
            ),
            axis=1
        )

        table = pd.concat([table, coords_df], axis=1, ignore_index=False)
        table.drop(columns=self.dom_id_cols, inplace=True)

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
        table['features'] = table[self.data_cols(table)].to_dict(orient='records')

        table = (
            table
            .groupby(self.event_id_cols, as_index=False)
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
        writer = LMDBWriter(table)
        writer.write(outfile, self.data_cols(table))

    @staticmethod
    def _apply_column_map(table: pd.DataFrame, mapping: dict) -> None:
        """
        Renames the columns of a DataFrame using the provided mapping.

        Args:
            table (pd.DataFrame): DataFrame to modify.
            mapping (dict): Mapping from original column names to new names.
        """
        table.rename(columns=mapping, inplace=True)

    def _normalize(self, table: pd.DataFrame, outfile: Union[str, Path]) -> pd.DataFrame:
        """
        Normalize the feature columns of a DataFrame and persist the scaling parameters.

        Args:
            table (pd.DataFrame):
                Input DataFrame containing raw feature columns.
            outfile (Union[str, Path]):
                File path where the JSON of fitted scaler parameters will be saved.

        Returns:
            pd.DataFrame:
                A copy of `table` with the specified feature columns normalized.
        """
        apply_log = [col for col in self.apply_log_scaling if col in table.columns]

        normalizer = Normalize(table)
        table = normalizer.transform(outfile, self.data_cols(table), apply_log)

        return table


class Normalize:
    """
    Utility class for normalizing and un-normalizing DataFrame columns using sklearn scalers.
    """
    def __init__(self, table: pd.DataFrame) -> None:
        self.table = table
        self._config = IGConfig.get()

    def transform(self, outfile: Union[str, Path], cols: list, log_cols: list) -> pd.DataFrame:
        """
        Normalize specified columns with z-score scaling and save parameters to JSON.

        Args:
            outfile: Path to write the scaler parameters JSON.
            cols: List of column names to normalize.
            log_cols: Columns to apply log10(x + 1) to.

        Returns:
            DataFrame with normalized columns.
        """
        outfile = Path(outfile)

        # apply log to specified columns
        self.table[log_cols] = np.log1p(self.table[log_cols])

        scaler = StandardScaler()
        table_scaled = self.table.copy()
        table_scaled[cols] = scaler.fit_transform(self.table[cols])

        log_mask = [col in log_cols for col in cols]

        params = {
            "columns": cols,
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "log": log_mask
        }

        with outfile.open("w") as file:
            json.dump(params, file, indent=2)

        return table_scaled

    def transform_from_saved_model(self, infile: Union[str, Path]) -> pd.DataFrame:
        """
        Normalize DataFrame columns using previously saved scaler parameters.

        Args:
            infile: Path to JSON file with scaler parameters.

        Returns:
            DataFrame with normalized columns.
        """
        infile = Path(infile)
        with infile.open("r") as file:
            params = json.load(file)

        table_norm = self.table.copy()
        for col, mu, sigma, log in zip(params["columns"], params["mean"], params["scale"], params["log"]):
            if log:
                table_norm[col] = np.log1p(table_norm[col])
            table_norm[col] = (self.table[col] - mu) / sigma

        return table_norm

    def inverse_transform(self, infile: Union[str, Path]) -> pd.DataFrame:
        """
        Revert normalized columns back to their original scale using saved parameters.

        Args:
            infile: Path to JSON file with scaler parameters.

        Returns:
            DataFrame with columns in original scale.
        """
        infile = Path(infile)
        with infile.open("r") as file:
            params = json.load(file)

        # only keep entries for columns that exist in self.table
        filtered_params = [
            (col, mu, sigma, log)
            for col, mu, sigma, log in zip(
                params["columns"],
                params["mean"],
                params["scale"],
                params["log"]
            )
            if col in self.table.columns
        ]

        if filtered_params:
            cols, means, scales, logs = zip(*filtered_params)
            params["columns"] = list(cols)
            params["mean"] = list(means)
            params["scale"] = list(scales)
            params["log"] = list(logs)
        else:
            # no matching columns → clear everything
            params["columns"] = []
            params["mean"] = []
            params["scale"] = []
            params["log"] = []

        table_denorm = self.table.copy()
        for col, mu, sigma, log in zip(params["columns"], params["mean"], params["scale"], params["log"]):
            table_denorm[col] = self.table[col] * sigma + mu
            if log:
                table_denorm[col] = np.expm1(table_denorm[col])

        return table_denorm

