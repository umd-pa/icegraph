# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from dataclasses import dataclass, field, asdict, fields
from typing import Iterable, List, Dict, Union, Optional, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd

ArrayF = npt.NDArray[np.float64]
ArrayI = npt.NDArray[np.int64]

@dataclass
class Statistics:
    """Mergeable per-column statistics (finite-only) for scalable normalization"""

    columns: List[str]
    n: ArrayI
    min: ArrayF
    max: ArrayF
    mean: ArrayF
    M2: ArrayF
    nan_count: ArrayI = field(repr=False)

    def _strip_numpy(self, obj):
        """Recursively convert numpy arrays in obj to Python lists."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._strip_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._strip_numpy(v) for v in obj]
        else:
            return obj

    @classmethod
    def from_dict(cls, d: dict) -> "Statistics":
        allowed = {f.name for f in fields(cls) if f.init}
        payload = {k: d[k] for k in allowed if k in d}

        # coerce lists -> numpy, if your dataclass expects arrays
        for k in ("n", "min", "max", "mean", "M2", "nan_count"):
            if k in payload and not isinstance(payload[k], np.ndarray):
                payload[k] = np.asarray(payload[k])

        missing = [name for name in allowed if name not in payload]
        if missing:
            raise TypeError(f"Missing required fields: {missing}")

        return Statistics(**payload)

    @classmethod
    def from_dense_array(cls, array: np.ndarray, columns: Sequence[str]) -> 'Statistics':
        """
            Create a Statistics object from a dense NumPy array and column names.

            Args:
                array: 2D NumPy array of shape (N, F).
                columns: Sequence of column names (length F).

            Returns:
                Statistics object.
            """
        if array.ndim != 2:
            raise ValueError("Array must be 2D (N, F).")
        if len(columns) != array.shape[1]:
            raise ValueError("Number of columns does not match array shape.")

        vals = np.asarray(array, dtype=np.float64)
        finite = np.isfinite(vals)

        n = finite.sum(axis=0, dtype=np.int64)

        # finite min/max with sentinels (+inf/-inf) when n==0
        with np.errstate(all="ignore"):
            col_min = np.where(n > 0, np.nanmin(np.where(finite, vals, np.nan), axis=0), np.inf)
            col_max = np.where(n > 0, np.nanmax(np.where(finite, vals, np.nan), axis=0), -np.inf)

        # finite mean
        sum_f = np.where(finite, vals, 0.0).sum(axis=0, dtype=np.float64)
        denom = np.maximum(n, 1).astype(np.float64)
        mean = sum_f / denom  # positions with n==0 will be ignored downstream

        diffs = np.where(finite, vals - mean, 0.0)
        M2 = (diffs * diffs).sum(axis=0, dtype=np.float64)

        nan_count = np.isnan(vals).sum(axis=0).astype(np.int64)

        return cls(
            columns=list(columns),
            n=n,
            min=col_min.astype(np.float64),
            max=col_max.astype(np.float64),
            mean=mean.astype(np.float64),
            M2=M2.astype(np.float64),
            nan_count=nan_count,
        )

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> 'Statistics':
        cols = list(df.columns)
        vals = df.to_numpy(dtype=np.float64, copy=False)
        finite = np.isfinite(vals)

        n = finite.sum(axis=0, dtype=np.int64)

        # finite min/max with sentinels (+inf/-inf) when n==0
        with np.errstate(all="ignore"):
            col_min = np.where(n > 0, np.nanmin(np.where(finite, vals, np.nan), axis=0), np.inf)
            col_max = np.where(n > 0, np.nanmax(np.where(finite, vals, np.nan), axis=0), -np.inf)

        # finite mean
        sum_f = np.where(finite, vals, 0.0).sum(axis=0, dtype=np.float64)
        denom = np.maximum(n, 1).astype(np.float64)
        mean = sum_f / denom  # positions with n==0 will be ignored downstream

        diffs = np.where(finite, vals - mean, 0.0)
        M2 = (diffs * diffs).sum(axis=0, dtype=np.float64)

        nan_count = df.isna().sum(axis=0).to_numpy(dtype=np.int64)

        return cls(
            columns=cols,
            n=n,
            min=col_min.astype(np.float64),
            max=col_max.astype(np.float64),
            mean=mean.astype(np.float64),
            M2=M2.astype(np.float64),
            nan_count=nan_count,
        )

    def to_dict(self, strip_np: bool = False) -> Dict[str, Union[List, np.ndarray]]:
        """
        Converts the statistics object to a dict.

        Args:
            strip_np (bool): If True, strips numpy from all internal values and returns a pure Python object. Defaults to False.
        """
        out = asdict(self) if not strip_np else self._strip_numpy(asdict(self))
        return out

    def aligned_to(self, target_columns: List[str]) -> 'Statistics':
        """Return a copy re-ordered to match target_columns (same set required)."""
        if self.columns == target_columns:
            return self

        # check for column mismatch
        if set(self.columns) != set(target_columns):
            missing = set(target_columns) - set(self.columns)
            extra = set(self.columns) - set(target_columns)
            raise ValueError(f"Column mismatch. Missing={missing}, Extra={extra}")

        idx = np.array([self.columns.index(c) for c in target_columns], dtype=np.int64)
        return Statistics(
            columns=list(target_columns),
            n=self.n[idx],
            min=self.min[idx],
            max=self.max[idx],
            mean=self.mean[idx],
            M2=self.M2[idx],
            nan_count=self.nan_count[idx],
        )

    def merge(self, other: 'Statistics') -> 'Statistics':
        """Merge two Stats (Welford/Chan). Columns are aligned to self.columns."""
        b = other.aligned_to(self.columns)
        n1, n2 = self.n, b.n
        n = n1 + n2
        denom = np.where(n == 0, 1, n).astype(np.float64)

        delta = b.mean - self.mean
        mean = self.mean + delta * (n2 / denom)
        M2 = self.M2 + b.M2 + (delta * delta) * (n1 * n2 / denom)

        return Statistics(
            columns=list(self.columns),
            n=n,
            min=np.minimum(self.min, b.min),
            max=np.maximum(self.max, b.max),
            mean=mean,
            M2=M2,
            nan_count=self.nan_count + b.nan_count,
        )

    @classmethod
    def merge_many(cls, stats: Iterable['Statistics']) -> 'Statistics':
        it = iter(stats)
        acc = next(it)
        for s in it:
            acc = acc.merge(s)
        return acc
