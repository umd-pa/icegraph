# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import cast, Optional, Union
from pathlib import Path
import json

import pandas as pd
from pandas import HDFStore
import numpy as np
import torch
from torch_cluster import knn_graph
from sklearn.preprocessing import StandardScaler

from icegraph.console import Console
from icegraph.config import IGConfig
from icegraph.console.streams import suppress_stderr
from .schemas import generate_vector_mapping
from icegraph.geometry import Detector
from icegraph.data.writers import LMDBWriter
from icegraph.pathutils import PathResolver, PathValidator
from .base.exceptions import ProcessorError

__all__ = ["FeatureProcessor", "Normalize"]


class FeatureProcessor:
    """
    Transforms an HDF5 file of DOM-level event data into a Lightning Memory-Mapped Database (LMDB)
    of compressed graph samples, with edge indices and features for GNN processing.

    This includes:
    - Reshaping DOM-level features
    - Mapping feature vectors
    - Grouping by event
    - Col-wise data normalization
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
        PathValidator.is_valid_file(self.infile)

        self._config: IGConfig = IGConfig.get()

        # get relevant columns from config
        self.dom_id_cols = self._config.internal_config.column_names.dom_id_columns
        self.event_id_cols = self._config.internal_config.column_names.event_id_columns
        self.dom_pos_cols = self._config.internal_config.column_names.dom_position_columns

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
        Console.banner("Feature Processor")
        Console.out(f"Building graph samples from raw data: {self.infile}")

        with Console.spinner():
            resolver = PathResolver(path=outfile, origin=self.infile, extension="lmdb", stage="transformer")
            outfile = resolver.resolve()

            feat_key = self._config.user_config.table_names.features
            truth_key = self._config.user_config.table_names.truth

            try:
                with HDFStore(self.infile, mode='r') as store:
                    for key in (feat_key, truth_key):
                        if key not in store:
                            raise ProcessorError(f"Missing HDF5 key '{key}' in {self.infile}.")

            except Exception as e:
                raise ProcessorError(f"Error accessing HDF5 keys: {e}")

            # Load data to DataFrames
            # IDE might complain these aren't DataFrames; they are.
            # Suppressing very loud HDF5 mismatched header warning
            with suppress_stderr():
                features_table = cast(pd.DataFrame, pd.read_hdf(
                    self.infile,
                    key=feat_key
                ))
                truth_table = cast(pd.DataFrame, pd.read_hdf(
                    self.infile,
                    key=truth_key
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
            event_id = tuple(row[c] for c in self.event_id_cols)
            features_dict = row['features']
            dom_positions = torch.tensor(
                [[f[c] for c in self.dom_pos_cols] for f in features_dict],
                dtype=torch.float
            )
            n = dom_positions.size(0)
            k_eff = min(max(n - 1, 0), 10)
            if k_eff <= 0:
                return pd.Series({'edge_index': [], 'edge_weight': []})

            edge_index = knn_graph(dom_positions, k=k_eff, loop=False)
            src, dst = edge_index
            distances = torch.norm(dom_positions[src] - dom_positions[dst], dim=1)
            if not torch.isfinite(distances).all():
                raise ProcessorError(f"Non-finite distances for event {event_id}")
            return pd.Series({'edge_index': edge_index.tolist(), 'edge_weight': distances.tolist()})

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
        feat_ids = set(map(tuple, features[self.event_id_cols].values))
        truth_ids = set(map(tuple, truth[self.event_id_cols].values))
        missing = feat_ids.symmetric_difference(truth_ids)
        if missing:
            raise ProcessorError(f"Event ID mismatch between features and truth: {missing}")

        assert len(features) == len(truth), "ID collision, please rerun feature extraction."

        return pd.merge(
            features,
            truth,
            on=self.event_id_cols,
            how='left'
        )

    def _reshape_features_table(self, table: pd.DataFrame) -> pd.DataFrame:
        """
        Reshapes the features table by pivoting ml_suite generated vector data.

        Args:
            table (pd.DataFrame): Input features table.

        Returns:
            pd.DataFrame: Reshaped features table.
        """
        if table is None or table.empty:
            raise ProcessorError("Input features table is empty or None")

        detector = Detector()

        # func for safely computing dom coords (avoiding any silent nans)
        def __safe_coords(row: pd.Series) -> pd.Series:
            dom_ids = tuple(row[c] for c in self.dom_id_cols)
            try:
                coords = detector.get_dom_coords(*dom_ids)

            except Exception as e:
                raise ProcessorError(f"Failed to get coords for DOM {dom_ids}: {e}")

            if (coords is None) or (len(coords) != len(self.dom_pos_cols)):
                raise ProcessorError(f"Invalid coords for DOM {dom_ids}: {coords}")

            return pd.Series(coords, index=self.dom_pos_cols)

        # Pivot the table
        pivot_col = "vector_index"
        value_col = "item"
        index_cols = [c for c in table.columns if c not in {pivot_col, value_col}]

        # quick data checks
        for col in (pivot_col, value_col):
            if col not in table.columns:
                raise ProcessorError(f"Missing expected column '{col}' in features table")

        if not any(c in index_cols for c in self.dom_id_cols + self.event_id_cols):
            raise ProcessorError("No DOM or event ID columns found for pivot index")

        reshaped = table.pivot_table(
            index=index_cols,
            columns=pivot_col,
            values=value_col,
            aggfunc="first"
        ).reset_index()
        if reshaped.empty:
            raise ProcessorError("Reshaped features table is empty after pivot")

        # Verify pivot created contiguous vector indices
        vector_indices = sorted([col for col in reshaped.columns if isinstance(col, int)])
        expected = list(range(min(vector_indices, default=0), max(vector_indices, default=-1) + 1))
        if vector_indices != expected:
            raise ProcessorError(f"Non-contiguous vector indices: found {vector_indices}, expected {expected}")

        coords_df = reshaped.apply(__safe_coords, axis=1)
        # Check for NaN or infinite in coordinates
        if coords_df.isnull().any().any() or not np.isfinite(coords_df.values).all():
            raise ProcessorError("Detected NaN or infinite values in DOM coordinates")

        final = pd.concat([reshaped, coords_df], axis=1)
        final.drop(columns=self.dom_id_cols, inplace=True)

        # confirm DOM position columns present
        for pos in self.dom_pos_cols:
            if pos not in final.columns:
                raise ProcessorError(f"Missing DOM position column '{pos}' after concatenation")

        return final

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

        if log_cols and (self.table[log_cols] < -1).any().any():
            bad = [c for c in log_cols if (self.table[c] < -1).any()]
            raise ProcessorError(f"Values < -1 in log columns: {bad}")

        # apply log to specified columns
        self.table[log_cols] = np.log1p(self.table[log_cols])

        scaler = StandardScaler()
        table_scaled = self.table.copy()
        table_scaled[cols] = scaler.fit_transform(self.table[cols])

        log_mask = [col in log_cols for col in cols]

        # Guard against zero scale (constant) columns
        zero_scale = [c for c, s in zip(cols, scaler.scale_) if s == 0]
        if zero_scale:
            raise ProcessorError(f"Constant columns detected with zero scale: {zero_scale}")

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
        PathValidator.is_valid_file(infile)

        with infile.open('r') as f:
            params = json.load(f)

        for c, s in zip(params['columns'], params['scale']):
            if s == 0:
                raise ProcessorError(f"Zero scale for column '{c}' in saved model")

        table_norm = self.table.copy()
        for col, mu, sigma, log in zip(params['columns'], params['mean'], params['scale'], params['log']):
            if log:
                table_norm[col] = np.log1p(table_norm[col])
            table_norm[col] = (table_norm[col] - mu) / sigma

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
        PathValidator.is_valid_file(infile)

        with infile.open("r") as file:
            params = json.load(file)

        filtered = [
            (c, m, s, l)
            for c, m, s, l in zip(params["columns"], params["mean"], params["scale"], params["log"])
            if c in self.table.columns
        ]

        if filtered:
            cols, means, scales, logs = zip(*filtered)
            params = {"columns": list(cols), "mean": list(means), "scale": list(scales), "log": list(logs)}
        else:
            params = {"columns": [], "mean": [], "scale": [], "log": []}

        table_denorm = self.table.copy()
        for col, mu, sigma, log in zip(params["columns"], params["mean"], params["scale"], params["log"]):
            table_denorm[col] = self.table[col] * sigma + mu
            if log:
                table_denorm[col] = np.expm1(table_denorm[col])

        return table_denorm

