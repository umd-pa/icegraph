# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

import json
from pathlib import Path
import os
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional, List, TYPE_CHECKING, Dict, Set, DefaultDict

import pandas as pd
import numpy as np
import torch
from torch_cluster import knn_graph
import threading

from .schemas import generate_vector_mapping
from icegraph.geometry import Detector
from .base.exceptions import ProcessorError
from .base import Processor
from icegraph.utils import Statistics

if TYPE_CHECKING:
    from icegraph.data.pipeline import Pipeline

__all__ = [
    "FeatureProcessor",
    "TruthProcessor",
    "EdgeProcessor",
    "StandardSplitAllocator",
    "StratifiedSplitAllocator",
    "StatisticsProcessor",
    "ClassNormalizer"
]


class HelperMixin:
    """Mixin for left merging dataframes."""

    # for linters
    _parent: Pipeline

    @staticmethod
    def merge_to_env(
            env: Pipeline.Envelope,
            df: pd.DataFrame,
    ) -> None:
        """
        Left merge two DataFrames on all shared column names.
        """
        if env.df.empty:
            env.df = df
            return

        common = [c for c in env.df.columns if c in df.columns]

        if common:
            right_keep = common + [c for c in df.columns if c not in env.df.columns]
            env.df = env.df.merge(df[right_keep], on=common, how="left")
        else:
            right_keep = [c for c in df.columns if c not in env.df.columns]
            env.df = env.df.merge(df[right_keep], left_index=True, right_index=True, how="left")

    def _load_from_hdf(self, env: Pipeline.Envelope, key: str) -> pd.DataFrame:
        with env.fh.lock_shared(), self._parent.HDF5_LOCK:
            df: pd.DataFrame = pd.read_hdf( # type: ignore
                env.fh.src, key=key
            )
        return df


class FeatureProcessor(Processor, HelperMixin):
    """
    Transforms an HDF5 file of DOM-level event data into a Lightning Memory-Mapped Database (LMDB)
    of compressed graph samples, with edge indices and features for GNN processing.
    """

    def __init__(self):
        # call to super
        super().__init__()

        # grab config values
        self.dom_id_cols = self._config.internal_config.column_names.dom_id_columns
        self.event_id_cols = self._config.internal_config.column_names.event_id_columns
        self.dom_pos_cols = self._config.internal_config.column_names.dom_position_columns
        self.apply_log_scaling_x = self._config.user_config.data.normalization.apply_log_scaling_x

        # get keys
        self.target_key = self._config.user_config.table_names.features

        self._detector_tls = threading.local()

    def _get_detector(self):
        det = getattr(self._detector_tls, "det", None)
        if det is None:
            det = Detector()
            self._detector_tls.det = det
        return det

    def _process(self, env: Pipeline.Envelope) -> Optional[Pipeline.Envelope]:
        # load required data from envelope
        df = self._load_from_hdf(env, self.target_key)

        # Run reshaping
        df = self._reshape_features_table(df)

        # Apply feature vector mapping
        vector_map = generate_vector_mapping()
        df = df.rename(columns=vector_map)

        # compress features by event, grab feature cols before compression
        feature_cols = self._get_feature_cols(df)
        df = self._compress(df)

        # register metadata to the envelope
        self._register_metadata(env, feature_cols)

        # merge this table to the main df and return
        self.merge_to_env(env, df)
        return env

    def _get_feature_cols(self, table: pd.DataFrame) -> List[str]:
        return [c for c in table.columns if c not in self.dom_id_cols + self.event_id_cols]

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
        det = self._get_detector()
        dom_ids_df = reshaped[self.dom_id_cols].drop_duplicates()
        coords = []
        for row in dom_ids_df.itertuples(index=False, name=None):
            try:
                xyz = det.get_dom_coords(*row)
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
        feature_cols = self._get_feature_cols(table)

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

    def _register_metadata(self, env: Pipeline.Envelope, feature_cols: List[str]) -> None:

        env.attrs["global"]["feature_names"] = feature_cols
        env.attrs["global"]["apply_log_scaling_x"] = self.apply_log_scaling_x


class TruthProcessor(Processor, HelperMixin):

    def __init__(self):
        # call to super
        super().__init__()

        # grab config values
        self.event_id_cols = self._config.internal_config.column_names.event_id_columns
        self.target_labels = self._config.user_config.data.target_labels
        self.apply_log_scaling_y = self._config.user_config.data.normalization.apply_log_scaling_y
        self.include_labels = self._config.user_config.data.include_labels

        # get keys
        self.target_key = self._config.user_config.table_names.truth

    def _process(self, env: Pipeline.Envelope) -> Optional[Pipeline.Envelope]:
        # load the required data
        df = self._load_from_hdf(env, self.target_key)

        # clean table if source hdf5 is fixed format and all cols were loaded
        truth_needed = list(set(self.event_id_cols + list(self.target_labels) + list(self.include_labels)))
        df = df[truth_needed]

        # register associated metadata
        self._register_metadata(env)

        # merge this table to the main df and return
        self.merge_to_env(env, df)
        return env

    def _register_metadata(self, env: Pipeline.Envelope) -> None:

        env.attrs["global"]["target_labels"] = self.target_labels
        env.attrs["global"]["include_labels"] = [label for label in self.include_labels if label not in self.target_labels]
        env.attrs["global"]["apply_log_scaling_y"] = self.apply_log_scaling_y


class EdgeProcessor(Processor, HelperMixin):

    PRE_REQS = [FeatureProcessor]

    def _process(self, env: Pipeline.Envelope) -> Optional[Pipeline.Envelope]:
        df = self._get_edge_index(env.df)

        # merge to envelope and return
        self.merge_to_env(env, df)
        return env

    @staticmethod
    def _get_edge_index(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute k-nearest neighbor graph edges and distances for each event's DOM features.
        """
        def compute_edges_from_dense(array: np.ndarray):
            if array is None or array.size == 0:
                return [], []
            pos_np = array[:, -3:].astype(np.float32, copy=False)  # last 3 = (x, y, z)

            # If any non-finite coords, drop those rows
            if not np.isfinite(pos_np).all():
                mask = np.isfinite(pos_np).all(axis=1)
                pos_np = pos_np[mask]
                if pos_np.size == 0:
                    return [], []

            pos = torch.from_numpy(pos_np)  # [N, 3]
            n = pos.size(0)
            k_eff = min(10, max(n - 1, 0))
            if k_eff == 0:
                return [], []

            edge_index = knn_graph(pos, k=k_eff, loop=False)
            src, dst = edge_index
            distances = torch.linalg.norm(pos[src] - pos[dst], dim=1)

            if not torch.isfinite(distances).all():
                raise ValueError("Non-finite edge distances encountered.")

            return edge_index.cpu().numpy().copy(), distances.cpu().numpy().copy()

        pairs = df["features"].map(compute_edges_from_dense)  # sequence of (edge_index, edge_weight)
        edge_index, edge_weight = zip(*pairs)  # unzip
        out_df = pd.DataFrame({"edge_index": edge_index, "edge_weight": edge_weight}, index=df.index)

        return out_df


class StandardSplitAllocator(Processor, HelperMixin):

    PRE_REQS = [TruthProcessor]

    def _process(self, env: Pipeline.Envelope) -> Optional[Pipeline.Envelope]:
        # grab weights and seed
        weights = self._config.user_config.data.splits.weights
        seed = self._config.user_config.training.seed
        N = len(env.df)

        p = np.array([weights[k] for k in ("train", "val", "test")], dtype=float)
        p /= p.sum()  # normalize

        rng = np.random.default_rng(seed)

        # target counts, round deterministically
        raw = p * N
        counts = np.floor(raw).astype(int)
        remainder = N - counts.sum()
        if remainder > 0:
            # give leftovers to the largest fractional parts; break ties randomly (but deterministically)
            frac = raw - np.floor(raw)
            jitter = rng.random(len(p)) * 1e-12
            order = np.argsort(-(frac + jitter))
            counts[order[:remainder]] += 1

        # enforce minimum of 1 for any split with p > 0 (if feasible)
        pos = p > 0
        need = (counts == 0) & pos
        k = int(need.sum())
        if k and N >= int(pos.sum()):
            # reassign from biggest-count splits until each needed split has 1
            donors = np.argsort(-counts)
            di = 0
            for s in np.where(need)[0]:
                # find a donor with count > 1
                while di < len(donors) and counts[donors[di]] <= 1:
                    di += 1
                if di >= len(donors):
                    break
                counts[donors[di]] -= 1
                counts[s] += 1

        # if some p==0, those splits are allowed to remain zero.
        assert counts.sum() == N, "Counts must sum to N."

        # create a splitmap with random row assignment
        indices = np.arange(N)
        rng.shuffle(indices)  # randomize which rows go to which split
        splitmap = np.empty(N, dtype=np.int64)
        start = 0
        for s, c in enumerate(counts):
            if c:
                splitmap[indices[start:start + c]] = s
            start += c

        # stash for downstream
        env.attrs["allocation"]["splitmap"] = splitmap

        return env


class StratifiedSplitAllocator(Processor, HelperMixin):

    """
    Implementation of deficit round-robin for online stratification.

    See: https://courses.cs.duke.edu/fall24/compsci514/readings/drr.pdf
    """

    PRE_REQS = [FeatureProcessor, TruthProcessor]

    @dataclass
    class _SplitTracker:
        split: int  # which split this tracker represents
        w: float  # normalized weight for this split
        K: int
        # per-class deficit for this split
        delta: Dict[object, float] = field(default_factory=dict)

        def ensure_class(self, c: object) -> None:
            if c not in self.delta:
                self.delta[c] = 0.0

        def update_for_class(self, assigned_split: int, c: object) -> None:
            self.ensure_class(c)
            self.delta[c] += self.w
            if assigned_split == self.split:
                self.delta[c] -= 1.0

    def _process(self, env: Pipeline.Envelope) -> Optional[Pipeline.Envelope]:
        # grab target labels and weights from config
        target_labels = self._config.user_config.data.target_labels
        weights_dict = self._config.user_config.data.splits.weights

        # get normalized weights for each split
        split_order = ["train", "val", "test"]
        w = np.array([weights_dict[k] for k in split_order], dtype=float)
        w = w / w.sum()

        # get total split count K
        K = len(w)

        # count number of samples n
        labels = env.df[target_labels].to_numpy()
        n = labels.size[0]

        # per-split trackers and per-class round-robin pointers
        trackers = [self._SplitTracker(split=i, w=float(w[i]), K=K) for i in range(K)]
        rr_ptr_by_class: Dict[object, int] = {}

        # online assignment
        out = np.empty(n, dtype=np.int8)

        for i, c in enumerate(labels):
            # ensure class state exists
            if c not in rr_ptr_by_class:
                rr_ptr_by_class[c] = 0
                for t in trackers:
                    t.ensure_class(c)

            # deficits for this class across splits
            deficits = np.array([t.delta[c] for t in trackers], dtype=float)
            # choose argmax(deficit) with round-robin tie-break
            mx = deficits.max()
            cand = np.flatnonzero(deficits >= mx - 1e-12)
            rr = rr_ptr_by_class[c]

            chosen = None
            for step in range(K):
                s = (rr + step) % K
                if s in cand:
                    chosen = s
                    break
            if chosen is None:  # very unlikely, fallback
                chosen = int(cand[0])

            out[i] = chosen

            # update deficits for all splits for this class
            for t in trackers:
                t.update_for_class(chosen, c)

            # advance RR pointer for this class
            rr_ptr_by_class[c] = (chosen + 1) % K

        # store result and return
        env.attrs["allocation"]["splitmap"] = out
        return env


class StatisticsProcessor(Processor, HelperMixin):

    PRE_REQS = [FeatureProcessor, (StandardSplitAllocator, StratifiedSplitAllocator)]

    def _process(self, env: Pipeline.Envelope) -> Optional[Pipeline.Envelope]:
        # grab necessary values from envelope
        feature_cols = env.attrs["global"]["feature_names"]
        splitmap = np.array(env.attrs["allocation"]["splitmap"])
        target_labels = env.attrs["global"]["target_labels"]
        df = env.df

        # get split labels from ints for human readability
        split_int_assignments = self._config.internal_config.split_int_assignments
        int_to_split_str: Dict[int, str] = dict(map(reversed, split_int_assignments.items()))

        # get feature and label stats for each split
        f_stats: Dict[str, Dict] = {}
        l_stats: Dict[str, Dict] = {}

        for split in range(3):
            split_name = int_to_split_str[split]
            sub_df = df[splitmap == split]

            # build stats from all dense arrays in split
            stats: Optional[Statistics] = None
            for array in sub_df["features"].to_numpy():
                s = Statistics.from_dense_array(array, feature_cols)
                stats = s if stats is None else stats.merge(s)

            # add to main dict as a dict
            f_stats[split_name] = stats.to_dict() if stats else {}

            l_stats[split_name] = (
                Statistics.from_dataframe(sub_df[target_labels]).to_dict() if not sub_df.empty else {}
            )

        env.attrs["stat"]["feature_stats"] = f_stats
        env.attrs["stat"]["label_stats"] = l_stats

        return env


class ClassNormalizer(Processor, HelperMixin):

    PRE_REQS = [TruthProcessor]

    _transfer_file: str = "_classmap.json"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # per-process cache of global classmap
        self._classmap: DefaultDict[str, List[str]] = defaultdict(list)

    def _process(self, env: Pipeline.Envelope) -> Optional[Pipeline.Envelope]:
        target_labels = env.attrs["global"]["target_labels"]

        local_diff: Dict[str, List[str]] = {}
        str_casted_series: Dict[str, pd.Series[str]] = {}
        unique_sets: Dict[str, Set[str]] = {}

        for label in target_labels:
            # grab all unique values in this batch
            str_casted_series[label] = env.df[label].astype(str)
            ordered_unique = pd.unique(str_casted_series[label])

            # stash for next loop
            unique_sets[label] = set(ordered_unique)

            existing = set(self._classmap[label])

            # check for diffs
            diff: List[str]
            if diff := [c for c in ordered_unique if c not in existing]:
                local_diff[label] = diff

        # sync the classmap with the global one if there are any diffs
        if local_diff:
            self._sync_classmap(local_diff)

        # apply mapping
        filtered_classmap: Dict[str, Dict[str, int]] = {}
        for label in target_labels:
            # build mapping order (IDs are indices into this list)
            cats = self._classmap[label]

            s = str_casted_series[label]
            codes = pd.Categorical(s, categories=cats, ordered=True).codes  # int array

            # fail loudly on unknowns
            if (codes == -1).any():
                missing = pd.unique(s[codes == -1]).tolist()
                raise ProcessorError(f"Unknown classes in '{label}': {missing}")

            max_code = int(codes.max(initial=0))
            env.df[label] = (
                codes.astype(np.uint8, copy=False)
                if max_code <= 255 else
                codes.astype(np.uint16, copy=False)
            )

            # minimal classmap for this batch
            filtered_classmap[label] = {cls: i for i, cls in enumerate(cats) if cls in unique_sets[label]}

        env.attrs["map"]["classmap"] = filtered_classmap

        return env

    def _sync_classmap(self, map_diff: Dict[str, List[str]]) -> None:
        # grab the temp dir and file paths
        temp_dir = self._parent.global_working_dir
        path = temp_dir / self._transfer_file

        # ensure file exists
        if not path.exists():
            with self._parent.global_tmpdir_obj.exclusive_write_lock(path):
                if not path.exists():
                    self._atomic_write_json(path, {})

        # grab file lock
        with self._parent.global_tmpdir_obj.exclusive_write_lock(path):
            gmap = self._read_json(path)

            # merge diffs preserving provided order, no duplicates
            for label, new_items in map_diff.items():
                lst = gmap.get(label, [])
                seen = set(lst)
                # keep order from the diff list
                additions = [x for x in new_items if x not in seen]
                if additions:
                    lst.extend(additions)
                    gmap[label] = lst

            # write atomically
            self._atomic_write_json(path, gmap)

            # refresh local cache from global
            self._classmap.clear()
            for k, v in gmap.items():
                self._classmap[k] = list(v)

    @staticmethod
    def _read_json(path: Path) -> Dict[str, List[str]]:
        try:
            with path.open("r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
        except json.JSONDecodeError:
            # extremely unlikely with atomic replace, retry once
            with path.open("r") as f:
                return json.load(f)

    @staticmethod
    def _atomic_write_json(path: Path, payload: Dict[str, List[str]]) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        os.replace(tmp, path)  # atomic on POSIX/Windows
        # best-effort fsync of the directory
        try:
            dfd = os.open(str(path.parent), os.O_RDONLY)
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)
        except OSError:
            pass
