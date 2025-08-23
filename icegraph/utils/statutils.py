# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from dataclasses import dataclass, field, asdict, fields
from typing import Iterable, List, Dict, Union, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd

ArrayF = npt.NDArray[np.float64]
ArrayI = npt.NDArray[np.int64]


@dataclass
class Statistics:
    """Mergeable per-column statistics (finite-only) for scalable normalization."""

    columns: List[str]

    # core finite-space moments
    n: ArrayI                 # count of finite values
    min: ArrayF
    max: ArrayF
    mean: ArrayF
    M2: ArrayF                # sum of squared deviations from mean (finite only)
    nan_count: ArrayI = field(repr=False)

    # magnitude stats for log-like scaling
    n_abs: ArrayI             # count of finite and |x|>0
    sum_log10_abs: ArrayF     # sum of log10(|x|) over finite and |x|>0

    # diagnostics
    n_zero: ArrayI            # count of exactly-zero values, irrespective of finiteness

    def _strip_numpy(self, obj):
        """Recursively convert numpy arrays in obj to Python lists."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: self._strip_numpy(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._strip_numpy(v) for v in obj]
        return obj

    @classmethod
    def from_dict(cls, d: dict) -> "Statistics":
        allowed = {f.name for f in fields(cls) if f.init}
        payload = {k: d[k] for k in allowed if k in d}

        # ints
        for k in ("n", "nan_count", "n_abs", "n_zero"):
            if k in payload and not isinstance(payload[k], np.ndarray):
                payload[k] = np.asarray(payload[k], dtype=np.int64)
        # floats
        for k in ("min", "max", "mean", "M2", "sum_log10_abs"):
            if k in payload and not isinstance(payload[k], np.ndarray):
                payload[k] = np.asarray(payload[k], dtype=np.float64)

        missing = [name for name in allowed if name not in payload]
        if missing:
            raise TypeError(f"Missing required fields: {missing}")

        return Statistics(**payload)

    @classmethod
    def _from_values(cls, vals: np.ndarray, columns: Sequence[str]) -> "Statistics":
        """Build stats from a 2D float64 array."""
        if vals.ndim != 2:
            raise ValueError("Array must be 2D (N, F).")
        if len(columns) != vals.shape[1]:
            raise ValueError("Number of columns does not match array shape.")

        vals = np.asarray(vals, dtype=np.float64)
        finite = np.isfinite(vals)

        # counts
        n = finite.sum(axis=0, dtype=np.int64)
        n_zero = (vals == 0.0).sum(axis=0, dtype=np.int64)

        # finite min/max with sentinels when n=0
        with np.errstate(all="ignore"):
            col_min = np.where(n > 0, np.nanmin(np.where(finite, vals, np.nan), axis=0), np.inf)
            col_max = np.where(n > 0, np.nanmax(np.where(finite, vals, np.nan), axis=0), -np.inf)

        # finite mean
        sum_f = np.where(finite, vals, 0.0).sum(axis=0, dtype=np.float64)
        denom = np.maximum(n, 1).astype(np.float64)
        mean = sum_f / denom  # columns with n==0 will be ignored downstream

        # M2 (sum of squared deviations) over finite entries
        diffs = np.where(finite, vals - mean, 0.0)
        M2 = (diffs * diffs).sum(axis=0, dtype=np.float64)

        # magnitude stats for asinh cofactor
        mask_abs = finite & (np.abs(vals) > 0.0)
        n_abs = mask_abs.sum(axis=0, dtype=np.int64)
        with np.errstate(divide="ignore", invalid="ignore"):
            sum_log10_abs = np.where(mask_abs, np.log10(np.abs(vals)), 0.0).sum(
                axis=0, dtype=np.float64
            )

        nan_count = np.isnan(vals).sum(axis=0).astype(np.int64)

        return cls(
            columns=list(columns),
            n=n,
            min=col_min.astype(np.float64),
            max=col_max.astype(np.float64),
            mean=mean.astype(np.float64),
            M2=M2.astype(np.float64),
            nan_count=nan_count,
            n_abs=n_abs,
            sum_log10_abs=sum_log10_abs.astype(np.float64),
            n_zero=n_zero,
        )

    @classmethod
    def from_dense_array(cls, array: np.ndarray, columns: Sequence[str]) -> "Statistics":
        vals = np.asarray(array, dtype=np.float64)
        return cls._from_values(vals, columns)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "Statistics":
        cols = list(df.columns)
        vals = df.to_numpy(dtype=np.float64, copy=False)
        return cls._from_values(vals, cols)

    def to_dict(self, strip_np: bool = False) -> Dict[str, Union[List, np.ndarray]]:
        """Convert to a dict."""
        out = asdict(self) if not strip_np else self._strip_numpy(asdict(self))
        return out

    def aligned_to(self, target_columns: List[str]) -> "Statistics":
        """Return a copy re-ordered to match target_columns (same set required)."""
        if self.columns == target_columns:
            return self

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
            n_abs=self.n_abs[idx],
            sum_log10_abs=self.sum_log10_abs[idx],
            n_zero=self.n_zero[idx],
        )

    def merge(self, other: "Statistics") -> "Statistics":
        """Merge two stats objects (Chan/Welford for moments; sums for magnitude stats)."""
        b = other.aligned_to(self.columns)

        n1, n2 = self.n, b.n
        n = n1 + n2

        # guard against division by zero in means update
        denom = np.where(n == 0, 1, n).astype(np.float64)

        # merge means
        delta = b.mean - self.mean
        mean = self.mean + delta * (n2 / denom)

        # merge M2 (Chan)
        M2 = self.M2 + b.M2 + (delta * delta) * (n1 * n2 / denom)

        # merge min/max, counts, magnitude logs
        out = Statistics(
            columns=list(self.columns),
            n=n,
            min=np.minimum(self.min, b.min),
            max=np.maximum(self.max, b.max),
            mean=mean,
            M2=M2,
            nan_count=self.nan_count + b.nan_count,
            n_abs=self.n_abs + b.n_abs,
            sum_log10_abs=self.sum_log10_abs + b.sum_log10_abs,
            n_zero=self.n_zero + b.n_zero,
        )
        return out

    @classmethod
    def merge_many(cls, stats: Iterable["Statistics"]) -> "Statistics":
        it = iter(stats)
        acc = next(it)
        for s in it:
            acc = acc.merge(s)
        return acc

    def stddev(self, unbiased: bool = True) -> ArrayF:
        """Per-column standard deviation in the original space."""
        denom = (self.n - 1) if unbiased else self.n
        denom = denom.astype(np.float64)
        out = np.full_like(self.M2, np.nan, dtype=np.float64)
        valid = denom > 0
        out[valid] = np.sqrt(self.M2[valid] / denom[valid])
        return out

    def asinh_cofactor(
            self,
            *,
            alpha: float = 1.0,
            eps: float = 1e-12,
            use_std_fallback: bool = True,
            unbiased_std: bool = False,
    ) -> ArrayF:
        """
        Compute per-column cofactor c for base-10 asinh scaling:

            y = asinh(x / c) / ln(10)
            x = c * sinh(y * ln(10))

        Args:
            alpha: scales the fallback (e.g., 1–3).
            eps:   minimum cofactor to avoid division-by-zero.
            use_std_fallback: use finite-space std as fallback if no positive magnitudes.
            unbiased_std: use unbiased std (n-1) for fallback if True, else population.

        Returns:
            ArrayF of shape (F,) with cofactors.
        """
        c = np.full_like(self.mean, np.nan, dtype=np.float64)

        # primary estimate from magnitude logs
        has_mag = self.n_abs > 0
        c[has_mag] = 10.0 ** (self.sum_log10_abs[has_mag] / self.n_abs[has_mag].astype(np.float64))

        # fallbacks where we have no |x|>0 observations
        none_mag = ~has_mag
        if np.any(none_mag):
            if use_std_fallback:
                sigma = self.stddev(unbiased=unbiased_std)
                fallback = np.maximum(eps, alpha * sigma)
            else:
                fallback = np.full_like(self.mean, max(eps, alpha * 1.0), dtype=np.float64)
            c[none_mag] = fallback[none_mag]

        # ensure strictly positive
        c = np.maximum(c, eps)
        return c
