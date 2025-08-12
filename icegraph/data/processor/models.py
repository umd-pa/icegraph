# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import cast, Optional, Union, Sequence, List, Dict
from pathlib import Path
import json

import pandas
import pandas as pd
from pandas import HDFStore
import numpy as np
import torch
from torch_cluster import knn_graph

from icegraph.console import Console
from icegraph.config import IGConfig
from icegraph.console.streams import suppress_stderr
from .schemas import generate_vector_mapping
from icegraph.geometry import Detector
from icegraph.data.writers import LMDBWriter
from icegraph.pathutils import PathResolver, PathValidator
from .base.exceptions import ProcessorError
from icegraph.utils import Statistics

__all__ = ["FeatureProcessor"]


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

    def __init__(self, source: Union[str, Path, Sequence[Union[str, Path]]]) -> None:
        """
        Initialize the data processor with an input file.

        Args:
            source (Union[str, Path, Sequence[Union[str, Path]]]): Path or sequence of paths to HDF5 files or a directory containing HDF5 files.

        """
        self._source = Path(source)
        self._file_paths: List[str] = PathResolver.normalize_sources(source, ".hdf5", use_str=True)

        if not self._file_paths:
            raise ProcessorError(
                f"No input files found matching extension '.hdf5' under {Console.source_repr(source)}"
            )

        self._config: IGConfig = IGConfig.get()

        # get relevant columns from config
        self.dom_id_cols = self._config.internal_config.column_names.dom_id_columns
        self.event_id_cols = self._config.internal_config.column_names.event_id_columns
        self.dom_pos_cols = self._config.internal_config.column_names.dom_position_columns

        # define a simple getter to extract data columns in a table and ignore IDs
        self.data_cols = lambda table: [c for c in table.columns if c not in self.dom_id_cols + self.event_id_cols]

        # get vector mapping
        self._vector_map = generate_vector_mapping()

    def __call__(self) -> Path:
        """
        Runs the sample building script. Saves the resulting samples to LMDB file(s).

        Returns:
            Path: Path to the output LMDB file directory.
        """
        return self.process()

    def process(self, outdir: Optional[Union[str, Path]] = None) -> Path:
        """
        Runs the sample building script. Saves the resulting samples to LMDB file(s).

        Returns:
            Path: Path to the output LMDB file directory.
        """
        source_repr = Console.source_repr(self._source)

        Console.banner("Feature Processor")
        Console.out(f"Building graph samples from source: {source_repr}")

        resolver = PathResolver(path=outdir, origin=None, extension="lmdb", stage="processor")
        outdir = resolver.resolve(return_dir=True)

        for infile in Console.progress_bar(self._file_paths):
            self._process_file(infile, outdir)

        Console.out("Processing complete!")

        return outdir

    def _process_file(self, infile: str, outdir: Path) -> Path:
        """Run the processing pipeline on one file."""
        feat_key = self._config.user_config.table_names.features
        truth_key = self._config.user_config.table_names.truth

        try:
            with suppress_stderr():
                with HDFStore(infile, mode='r') as store:
                    for key in (feat_key, truth_key):
                        if key not in store:
                            raise ProcessorError(f"Missing HDF5 key '{key}' in {infile}.")

        except Exception as e:
            raise ProcessorError(f"Error accessing HDF5 keys: {e}")

        # Load data to DataFrames
        # IDE might complain these aren't DataFrames; they are.
        # Suppressing very loud HDF5 mismatched header warning
        with suppress_stderr():
            features_table = cast(pd.DataFrame, pd.read_hdf(
                infile,
                key=feat_key
            ))
            truth_table = cast(pd.DataFrame, pd.read_hdf(
                infile,
                key=truth_key
            ))

        # Run reshaping
        features_table = self._reshape_features_table(features_table)

        # Apply feature vector mapping
        self._apply_column_map(features_table, self._vector_map)

        # compress features by event
        features_table = self._compress(features_table)

        # merge the truth and features tables and calculate edge indices
        table = self._merge_tables(features_table, truth_table)
        table = self._append_edge_index(table)

        # export to lmdb
        outfile = outdir / Path(infile).with_suffix(".lmdb").name
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
        def compute_edges_from_dense(X: np.ndarray):
            if X is None or X.size == 0:
                return [], []
            pos_np = X[:, -3:].astype(np.float32, copy=False)  # last 3 = (x, y, z)

            # If any non-finite coords, drop those rows
            if not np.isfinite(pos_np).all():
                mask = np.isfinite(pos_np).all(axis=1)
                pos_np = pos_np[mask]
                if pos_np.size == 0:
                    return [], []

            pos = torch.from_numpy(pos_np)  # [N, 3]
            n = pos.size(0)
            k_eff = min(10, max(n - 1, 0))  # <= N-1 neighbors; 0 if n<=1
            if k_eff == 0:
                return [], []

            edge_index = knn_graph(pos, k=k_eff, loop=False)  # [2, E], directed
            src, dst = edge_index
            distances = torch.linalg.norm(pos[src] - pos[dst], dim=1)

            if not torch.isfinite(distances).all():
                raise ValueError("Non-finite edge distances encountered.")

            return edge_index.tolist(), distances.tolist()

        out = table["features"].apply(
            lambda X: pd.Series(compute_edges_from_dense(X), index=["edge_index", "edge_weight"])
        )
        table[["edge_index", "edge_weight"]] = out
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

        assert len(features) == len(truth), "Potential ID collision, please rerun feature extraction."

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
        Group DOM-level features into a dense array for each event and return an event-level table.

        This method converts rows corresponding to individual DOMs into a single dense NumPy array
        of DOM features per event. The resulting table has one row per event.

        Args:
            table (pd.DataFrame): The DOM-level feature table.

        Returns:
            pd.DataFrame: The event-level feature table with one dense array of features per event.
        """
        feature_cols = self.data_cols(table)

        # one dense row-array per DOM hit
        table = table.copy()
        table["features"] = list(table[feature_cols].to_numpy(dtype=np.float32))

        out = (
            table
            .groupby(self.event_id_cols, as_index=False, sort=False)
            .agg(features=("features", lambda x: np.vstack(x)))
        )

        return out

    def _build_metadata(self, table: pd.DataFrame) -> Dict:
        """
        Build the metadata to write to the LMDB file.

        Args:
            table (pd.DataFrame): Data that will be written to the LMDB.
        """
        # get feature and excluded columns
        feature_cols = list(self._vector_map.values()) + self.dom_pos_cols
        exclude_cols = ["edge_index", "edge_weight"]

        # drop excluded cols
        table = table.drop(columns=exclude_cols)

        # collect stats
        partial_f_stats: List[Statistics] = []
        for array in table["features"].to_numpy():
            partial_f_stats.append(Statistics.from_dense_array(array, feature_cols))

        f_stats = Statistics.merge_many(partial_f_stats)
        t_stats = Statistics.from_dataframe(table.drop(columns=["features"]))

        metadata = {
            "f_stats": f_stats.to_dict(),
            "t_stats": t_stats.to_dict(),
            "stats_policy": {
                "exclude_cols": exclude_cols
            },
            "schema": {
                "feature_cols": feature_cols
            }
        }
        return metadata

    def _to_lmdb(self, table: pd.DataFrame, outfile: Union[str, Path]) -> None:
        """
        Writes the given DataFrame to an LMDB file.

        Args:
            table (pd.DataFrame): Data to write.
            outfile (Union[str, Path]): Output file path.
        """
        include_cols = self.data_cols(table)
        metadata = self._build_metadata(table[include_cols])

        with LMDBWriter(outfile, verbose=False) as file:
            file.write_metadata(metadata)
            file.write(table, include_cols)

    @staticmethod
    def _apply_column_map(table: pd.DataFrame, mapping: dict) -> None:
        """
        Renames the columns of a DataFrame using the provided mapping.

        Args:
            table (pd.DataFrame): DataFrame to modify.
            mapping (dict): Mapping from original column names to new names.
        """
        table.rename(columns=mapping, inplace=True)
