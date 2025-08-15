# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from typing import cast, Optional, Union, Sequence, List, Dict
from pathlib import Path

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
from icegraph.utils import Statistics, stable_hash_cbor

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

        # grab config and detector
        self._config: IGConfig = IGConfig.get()
        self._detector = Detector()

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
        # NOTE: Read only required columns when possible to reduce I/O; for 'fixed' HDF this may be ignored by pandas (still safe).
        feat_needed = list(set(self.dom_id_cols + self.event_id_cols + ["vector_index", "item"]))
        truth_needed = self.event_id_cols + list(getattr(self._config.user_config.data, "target_labels", []))
        with suppress_stderr():
            features_table = cast(pd.DataFrame, pd.read_hdf(
                infile,
                key=feat_key,
                columns=feat_needed
            ))
            truth_table = cast(pd.DataFrame, pd.read_hdf(
                infile,
                key=truth_key,
                columns=truth_needed
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

    @staticmethod
    def _append_edge_index(table: pd.DataFrame) -> pd.DataFrame:
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
        feats_col = table["features"].values
        edge_index_list: List[list] = []
        edge_weight_list: List[list] = []

        for X in feats_col:
            if X is None or X.size == 0:
                edge_index_list.append([])
                edge_weight_list.append([])
                continue

            pos_np = X[:, -3:].astype(np.float32, copy=False)  # last 3 = (x, y, z)

            # If any non-finite coords, drop those rows
            finite = np.isfinite(pos_np).all(axis=1)
            if not finite.all():
                pos_np = pos_np[finite]
                if pos_np.size == 0:
                    edge_index_list.append([])
                    edge_weight_list.append([])
                    continue

            n = pos_np.shape[0]
            k_eff = min(10, max(n - 1, 0))  # <= N-1 neighbors; 0 if n<=1
            if k_eff == 0:
                edge_index_list.append([])
                edge_weight_list.append([])
                continue

            with torch.no_grad():
                pos = torch.from_numpy(pos_np)  # [N, 3]
                edge_index = knn_graph(pos, k=k_eff, loop=False)  # [2, E], directed
                src, dst = edge_index
                distances = torch.linalg.norm(pos[src] - pos[dst], dim=1)

            if not torch.isfinite(distances).all():
                raise ValueError("Non-finite edge distances encountered.")

            edge_index_list.append(edge_index.tolist())
            edge_weight_list.append(distances.tolist())

        table = table.copy()
        table["edge_index"] = edge_index_list
        table["edge_weight"] = edge_weight_list
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

        # Equivalent to pivot_table but faster
        dedup = (
            table
            .sort_values(index_cols + [pivot_col], kind="mergesort")
            .drop_duplicates(subset=index_cols + [pivot_col], keep="first")
        )
        reshaped = (
            dedup
            .set_index(index_cols + [pivot_col])[value_col]
            .unstack(pivot_col)
            .reset_index()
        )
        if reshaped.empty:
            raise ProcessorError("Reshaped features table is empty after pivot")

        # Verify pivot created contiguous vector indices
        vector_indices = sorted([col for col in reshaped.columns if isinstance(col, int)])
        expected = list(range(min(vector_indices, default=0), max(vector_indices, default=-1) + 1))
        if vector_indices != expected:
            raise ProcessorError(f"Non-contiguous vector indices: found {vector_indices}, expected {expected}")

        # Vectorized coordinates join instead of per-row apply
        dom_ids_df = reshaped[self.dom_id_cols].drop_duplicates()
        coords = []
        for row in dom_ids_df.itertuples(index=False, name=None):
            try:
                xyz = self._detector.get_dom_coords(*row)
            except Exception as e:
                raise ProcessorError(f"Failed to get coords for DOM {row}: {e}")
            if (xyz is None) or (len(xyz) != len(self.dom_pos_cols)):
                raise ProcessorError(f"Invalid coords for DOM {row}: {xyz}")
            coords.append(xyz)

        coords_df = dom_ids_df.copy()
        for i, col in enumerate(self.dom_pos_cols):
            coords_df[col] = [c[i] for c in coords]

        final = reshaped.merge(coords_df, on=self.dom_id_cols, how="left")
        # Check for NaN or infinite in coordinates
        if final[self.dom_pos_cols].isnull().any().any() or not np.isfinite(final[self.dom_pos_cols].to_numpy()).all():
            raise ProcessorError("Detected NaN or infinite values in DOM coordinates")

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
        """
        feature_cols = self.data_cols(table)

        # one dense row-array per DOM hit
        feats = table[feature_cols].to_numpy(dtype=np.float32, copy=False)

        # Build event key codes
        key_arrays = [table[c].to_numpy(copy=False) for c in self.event_id_cols]
        mi = pd.MultiIndex.from_arrays(key_arrays, names=self.event_id_cols)
        codes, uniques = pd.factorize(mi, sort=False)

        order = np.argsort(codes, kind="mergesort")
        codes_sorted = codes[order]
        feats_sorted = feats[order]

        # Find group boundaries
        change = np.empty(len(codes_sorted), dtype=bool)
        change[0] = True
        change[1:] = codes_sorted[1:] != codes_sorted[:-1]
        start_idx = np.flatnonzero(change)
        end_idx = np.r_[start_idx[1:], len(codes_sorted)]

        arrays = [feats_sorted[s:e] for s, e in zip(start_idx, end_idx)]

        # Materialize event id columns (one row per group) in first-appearance order
        out = pd.DataFrame({name: np.asarray(uniques.get_level_values(i)) for i, name in enumerate(self.event_id_cols)})
        out["features"] = arrays

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
        f_stats: Optional[Statistics] = None
        for array in table["features"].to_numpy():
            s = Statistics.from_dense_array(array, feature_cols)
            f_stats = s if f_stats is None else f_stats.merge(s)
        if f_stats is None:
            f_stats = Statistics.from_dense_array(np.zeros((0, len(feature_cols)), dtype=np.float32), feature_cols)

        # grab statistics
        t_stats = Statistics.from_dataframe(table.drop(columns=["features"]))

        # load the ml suite config
        ml_suite_config = self._config.ml_suite_config

        # grab the target labels and log scaling mask from config
        target_labels = self._config.user_config.data.target_labels
        apply_log_scaling = self._config.user_config.data.normalization.apply_log_scaling

        metadata = {
            "f_stats": f_stats.to_dict(),
            "t_stats": t_stats.to_dict(),
            "metadata": {
                "feature_names": feature_cols,
                "target_labels": target_labels,
                "apply_log_scaling": apply_log_scaling,
                "config": ml_suite_config,
                "CBOR_canonical_blake2b": stable_hash_cbor(ml_suite_config)
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
