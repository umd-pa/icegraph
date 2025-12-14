# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import List, Dict, Union, Callable, Any, Tuple, TypeVar, TypeAlias, Literal
import functools
from collections.abc import Iterable

import numpy as np
import numpy.typing as npt

# type definitions
ArrayF: TypeAlias = npt.NDArray[np.float64]

StatStruct: TypeAlias = Dict[Literal["columns", "stats"], Union[List[str], List[List[float]]]]
TStats = TypeVar("TStats", bound="Statistics")

F = TypeVar("F", bound=Callable[..., Any])

# wrappers
def _guarded(func: F) -> F:
    """Decorator that raises if the instance's internal statistics array is uninitialized."""

    @functools.wraps(func)
    def inner(self, *args, **kwargs) -> Any:
        if getattr(self, "_stats", None) is None:
            raise ValueError("Internal statistics array has not been initialized.")
        return func(self, *args, **kwargs)

    return inner

def generate_stat_getters(cls: TStats) -> TStats:
    """Class decorator that adds guarded, read-only properties for each row in FIELDS."""

    def _create_getter(idx) -> Callable[..., np.ndarray]:
        @_guarded
        def getter(self) -> np.ndarray:
            return self._stats[idx]

        return getter

    # dynamically add getters: self.mean, self.maximum, ...
    for name, idx in cls.FIELDS.items():
        setattr(cls, name, property(_create_getter(idx)))

    return cls


@generate_stat_getters
class Statistics:
    """
    Mergeable per-column statistics (finite-only) for scalable normalization.

    The internal representation is a dense 2D array of shape (R, F), where R is
    the number of tracked statistics (rows corresponding to FIELDS) and F is
    the number of feature columns. This class is used to compute stats for individual
    dataset shards during data processing, and for determining global stats during
    training for normalization.
    """

    # DO NOT CHANGE THESE UNLESS YOU KNOW EXACTLY WHAT YOU ARE DOING
    # backwards compatibility is very important to maintain here and is easy to break
    N               = 0  # number of finite values
    MINIMUM         = 1  # min value
    MAXIMUM         = 2  # max value
    MEAN            = 3  # mean value
    M2              = 4  # M2 aggregates the squared distance from the mean (see Welford algorithm)
    NAN_COUNT       = 5  # number of NaN values
    N_ABS           = 6  # count of finite and |x|>0
    SUM_LOG10_ABS   = 7  # sum of log10(|x|) over finite and |x|>0
    N_ZERO          = 8  # number of 0s for diagnostics

    # auto-generate FIELDS from constant attrs defined above
    FIELDS: Dict[str, int] = {
        name.lower(): value
        for name, value in vars().items()
        if name.isupper() and isinstance(value, int)
    }

    # enforce contiguous indices for FIELDS
    assert set(FIELDS.values()) == set(range(len(FIELDS)))

    _columns: List[str]

    # internal dense array
    # rows (in order): n, min, max, mean, M2, nan_count, n_abs, sum_log10_abs, n_zero
    _stats: ArrayF

    # rowwise ops
    @staticmethod
    def _merge_sum(stats: List[Statistics], idx: int, _: ArrayF) -> ArrayF:
        """
        Merge a single statistic row by summation across instances.

        Args:
            stats: List of Statistics instances to merge.
            idx:   Row index into each instance's `_stats` array.
            _:     Accumulator array (unused for this merge operation).
        """
        arrays = [stat[idx] for stat in stats]
        return np.add.reduce(arrays, axis=0)

    @staticmethod
    def _merge_min(stats: List[Statistics], idx: int, _: ArrayF) -> ArrayF:
        """
        Merge a single statistic row by taking the elementwise minimum.

        Args:
            stats: List of Statistics instances to merge.
            idx:   Row index into each instance's `_stats` array.
            _:     Accumulator array (unused for this merge operation).
        """
        arrays = [stat[idx] for stat in stats]
        return np.minimum.reduce(arrays, axis=0)

    @staticmethod
    def _merge_max(stats: List[Statistics], idx: int, _: ArrayF) -> ArrayF:
        """
        Merge a single statistic row by taking the elementwise maximum.

        Args:
            stats: List of Statistics instances to merge.
            idx:   Row index into each instance's `_stats` array.
            _:     Accumulator array (unused for this merge operation).
        """
        arrays = [stat[idx] for stat in stats]
        return np.maximum.reduce(arrays, axis=0)

    @classmethod
    def _merge_mean(cls, stats: List[Statistics], idx: int, _: ArrayF) -> ArrayF:
        """
        Merge per-column means using count-weighted averaging.

        Args:
            stats: List of Statistics instances to merge.
            idx:   Row index corresponding to the mean (MEAN).
            _:     Accumulator array (unused for this merge operation).
        """
        # stack into 2D arrays of shape (num_stats, num_features)
        arrays = [stat[idx] for stat in stats]
        weights = [stat[cls.N] for stat in stats]

        stacked_arrays = np.stack(arrays, axis=0)
        stacked_weights = np.stack(weights, axis=0)

        weighted_sum = np.sum(stacked_weights * stacked_arrays, axis=0)
        total_weight = np.sum(stacked_weights, axis=0)

        return np.divide(
            weighted_sum,
            total_weight,
            out=np.zeros_like(weighted_sum),
            where=total_weight > 0,
        )

    @classmethod
    def _merge_m2(cls, stats: List[Statistics], idx: int, accumulator: ArrayF) -> ArrayF:
        """
        Merge per-column M2 terms using the parallel Welford algorithm.

        Args:
            stats:        List of Statistics instances to merge.
            idx:          Row index corresponding to M2.
            accumulator:  Partially-filled accumulator; its MEAN row must
                          already contain the merged means.
        """
        arrays = [stat[idx] for stat in stats]
        weights = [stat[cls.N] for stat in stats]
        means = [stat[cls.MEAN] for stat in stats]

        stacked_arrays = np.stack(arrays, axis=0)  # shape (k, F)
        stacked_weights = np.stack(weights, axis=0)  # shape (k, F)
        stacked_means = np.stack(means, axis=0)  # shape (k, F)

        # grab precomputed merged mean from the accumulator
        total_mean = accumulator[cls.MEAN]

        # merged M2
        merged_arrays = stacked_arrays.sum(axis=0) + (stacked_weights * (stacked_means - total_mean) ** 2).sum(axis=0)

        return merged_arrays

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the shape of the internal statistics array (rows, columns)."""
        return self._stats.shape

    @property
    def columns(self) -> List[str]:
        """Return the list of column names corresponding to feature indices."""
        return self._columns

    def __getitem__(self, item: Any) -> ArrayF:
        """
        Return a slice or row of the internal statistics array.

        Args:
            item: Index or slice applied to the first axis of `_stats`.
        """
        return self._stats[item]

    def __iter__(self) -> Iterable[ArrayF]:
        """Iterate over rows of the internal statistics array."""
        return iter(self._stats)

    def __add__(self, other: Statistics) -> Statistics:
        """Addition of two Statistics objects yields a single merged instance."""
        return self.merge([self, other])

    def __radd__(self, other: Any) -> Statistics:
        """Right-hand addition for compatibility with `sum`."""
        # support sum([...]) which starts with 0
        if other == 0:
            return self

        # delegate to __add__
        return self.__add__(other)

    def __reduce__(self) -> Tuple[Callable[..., Any], Tuple[StatStruct]]:
        struct = self.to_struct()
        # When unpickling, pickle will call:
        # Statistics.load_struct(struct)
        # which will fully rebuild the instance
        return self.__class__.load_struct, (struct,)

    @classmethod
    def merge(cls, stats: Iterable[Statistics]) -> Statistics:
        """
        Merge two or more Statistics instances into one.

        Combines per-column counts, extrema, means, M2 terms, and
        related diagnostics across multiple instances, assuming they share the
        same internal shape and column ordering. Useful during training
        for computation of global stats.

        Args:
            stats: Iterable of Statistics objects to merge.

        Raises:
            TypeError:  If `stats` is not an iterable of Statistics instances.
            ValueError: If `stats` is empty or the shapes/columns differ.
        """
        # verify types
        type_error = TypeError("Argument 'stats' must be an iterable over Statistics objects.")
        if not isinstance(stats, Iterable) or isinstance(stats, (str, bytes)):
            raise type_error

        # coerce to list
        stats = list(stats)

        # break out if empty stats list
        if not stats:
            raise ValueError("Cannot merge an empty collection of Statistics.")

        # all items in collection need to be stat objects
        if any(not isinstance(s, cls) for s in stats):
            raise type_error

        # break out if only one object passed
        if len(stats) == 1:
            return stats[0]

        # assert consistent shapes and columns
        array_shape = stats[0].shape
        array_cols = stats[0].columns
        if any(a.shape != array_shape for a in stats):
            raise ValueError(f"Array shapes differ: expected {array_shape}, got {[a.shape for a in stats]}")
        if any(a.columns != array_cols for a in stats):
            raise ValueError(f"Array columns differ: expected {array_cols}, got {[a.columns for a in stats]}")

        # DO NOT CHANGE THIS ORDERING UNLESS YOU KNOW EXACTLY WHAT YOU ARE DOING
        methods: List[Callable[..., ArrayF]] = [
            cls._merge_sum,  # 0: N
            cls._merge_min,  # 1: MINIMUM
            cls._merge_max,  # 2: MAXIMUM
            cls._merge_mean, # 3: MEAN
            cls._merge_m2,   # 4: M2
            cls._merge_sum,  # 5: NAN_COUNT
            cls._merge_sum,  # 6: N_ABS
            cls._merge_sum,  # 7: SUM_LOG10_ABS
            cls._merge_sum,  # 8: N_ZERO
        ]

        # prevent drift
        assert len(methods) == len(cls.FIELDS)

        # build new stat array
        accumulator: ArrayF = np.zeros(array_shape)

        # iterate over each field and apply row op
        for idx, method in enumerate(methods):
            accumulator[idx] = method(stats, idx, accumulator)

        return cls.load_array(accumulator, array_cols)

    @classmethod
    def load_array(cls, stats_array: ArrayF, columns: List[str]) -> Statistics:
        """
        Construct a Statistics instance from a raw stats array and column names.

        This is a low-level constructor that bypasses `from_dense_array`. Generally, users
        should not be using this method and should instead be loading from a struct (for
        raw data), or building from a dense array.

        Args:
            stats_array: 2D array of shape (R, F) containing per-row statistics.
            columns:     List of length F with column names.
        """
        # instantiate
        instance = cls.__new__(cls)

        # populate
        instance._stats = stats_array
        instance._columns = columns

        # return
        return instance

    @classmethod
    def from_dense_array(cls, array: np.ndarray, columns: List[str]) -> Statistics:
        """
        Compute all statistics from a dense feature array and return a Statistics instance.

        The input array is treated as shape (N, F), where N is the number of
        samples and F is the number of feature columns. This is designed to directly
        accept data formatted by the FeatureProcessor during processing.

        Args:
            array:    2D NumPy array of shape (N, F) containing feature values.
            columns:  List of length F with column names corresponding to array
                      columns.

        Raises:
            ValueError: If `array` is not 2D or mismatch between column list and array columns.
        """
        # assertions
        if array.ndim != 2:
            raise ValueError("Array must be 2D (N, F).")
        if len(columns) != array.shape[1]:
            raise ValueError("Number of columns does not match array shape.")

        # init empty stat array of correct shape
        stats_array: ArrayF = np.zeros([len(cls.FIELDS), len(columns)])

        # convert to float64 and build masks
        f64_array: ArrayF = np.asarray(array, dtype=np.float64)
        finite_mask: npt.NDArray[np.bool_] = np.isfinite(f64_array)
        zero_mask: npt.NDArray[np.bool_] = (f64_array == 0.0)

        # compute counts and number of zeros
        stats_array[cls.N] = finite_mask.sum(axis=0, dtype=np.float64)
        stats_array[cls.N_ZERO] = zero_mask.sum(axis=0, dtype=np.float64)

        # build the nonzero finite mask
        non_zero_count_mask: npt.NDArray[np.bool_] = stats_array[cls.N] > 0

        # get the min and max (set to absurd values if data is missing so it is obvious downstream)
        with np.errstate(all="ignore"):
            stats_array[cls.MINIMUM] = np.where(
                non_zero_count_mask, np.nanmin(np.where(finite_mask, f64_array, np.nan), axis=0), np.inf
            )
            stats_array[cls.MAXIMUM] = np.where(
                non_zero_count_mask, np.nanmax(np.where(finite_mask, f64_array, np.nan), axis=0), -np.inf
            )

        # calculate finite mean
        sums = np.where(finite_mask, f64_array, 0.0).sum(axis=0, dtype=np.float64)
        counts = np.maximum(stats_array[cls.N], 1).astype(np.float64)
        stats_array[cls.MEAN] = sums / counts  # columns with n==0 will be ignored downstream

        # magnitude stats for asinh cofactor
        non_zero_finite_mask = finite_mask & (np.abs(f64_array) > 0.0)
        stats_array[cls.N_ABS] = non_zero_finite_mask.sum(axis=0, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            stats_array[cls.SUM_LOG10_ABS] = np.where(non_zero_finite_mask, np.log10(np.abs(f64_array)), 0.0).sum(
                axis=0, dtype=np.float64
            )

        # M2 (sum of squared deviations) over finite entries
        diffs = np.where(finite_mask, f64_array - stats_array[cls.MEAN], 0.0)
        stats_array[cls.M2] = (diffs * diffs).sum(axis=0, dtype=np.float64)

        # NaN count for diagnostics
        stats_array[cls.NAN_COUNT] = np.isnan(f64_array).sum(axis=0).astype(np.float64)

        return cls.load_array(stats_array, columns)

    def to_list(self) -> List[List[float]]:
        """Return a nested list representation of the internal statistics array."""
        return self._stats.tolist()

    def to_struct(self) -> StatStruct:
        """Serialize the Statistics instance into a JSON-friendly dict structure."""
        return {
            "columns": self._columns,
            "stats": self.to_list()
        }

    @classmethod
    def load_struct(cls, struct: StatStruct) -> Statistics:
        """
        Construct a Statistics instance from a serialized struct. Inverse method of `to_struct()`.

        Args:
            struct: A dictionary with keys "columns" and "stats", same format as produced
                    by `to_struct()`.

        Returns:
            A new Statistics instance populated from the serialized contents.
        """
        array = np.asarray(struct["stats"], dtype=np.float64)
        return cls.load_array(array, struct["columns"])

    def aligned_to(self, columns: List[str]) -> None:
        """
        Reorder internal columns and stats in-place to match a new column order.

        This method only permits reordering of columns; it will raise if the
        requested column set differs (aside from ordering) from the existing
        internal column set.

        Args:
            columns: Desired column ordering, containing exactly the same names as `self.columns`.

        Raises:
            ValueError: If either the current or requested column list contains duplicates.
            AssertionError: If the requested columns do not match the existing columns as an unordered set.
        """
        # ensure new column list is only reordering
        if len(set(self._columns)) != len(self._columns):
            raise ValueError("Internal column list contains duplicates.")
        if len(set(columns)) != len(columns):
            raise ValueError("Requested column list contains duplicates.")
        assert sorted(columns) == sorted(self._columns), "Columns do not match."

        # build fast lookup map
        col_to_idx = {col: i for i, col in enumerate(self._columns)}

        # compute index order
        index_map = [col_to_idx[col] for col in columns]

        # reorder and update
        self._stats = self._stats[:, index_map]
        self._columns = list(columns)

    # convenience methods

    @_guarded
    def stddev(self, unbiased: bool = True) -> ArrayF:
        """
        Compute per-column standard deviation.

        Args:
            unbiased: If True, use the unbiased estimator with denominator (n-1),
                      otherwise use the population estimator with denominator n.

        Returns:
            1D array of per-column standard deviations. Columns with insufficient
            counts (e.g. n < 2 for unbiased) are set to NaN.
        """
        # pull from stats array
        n = self._stats[self.N]
        m2 = self._stats[self.M2]

        # calculate columnwise stddev
        denom = ((n - 1) if unbiased else n).astype(np.float64)
        out = np.full_like(m2, np.nan, dtype=np.float64)
        valid = denom > 0
        out[valid] = np.sqrt(m2[valid] / denom[valid])
        return out

    @_guarded
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

        The primary estimate is based on the geometric mean of |x| over nonzero
        finite magnitudes. For columns with no |x| > 0, an optional stddev-based fallback is used.

        Args:
            alpha:          Scaling factor applied to the standard-deviation
                            fallback (e.g. 1–3).
            eps:            Minimum allowed cofactor to avoid division-by-zero.
            use_std_fallback:
                            If True, use finite-space stddev as fallback when
                            there are no positive magnitudes; otherwise a fixed
                            fallback is used.
            unbiased_std:   If True, use unbiased stddev (n-1) for fallback;
                            otherwise use the population stddev.

        Returns:
            1D ArrayF of shape (F,) with per-column asinh scaling cofactors.
        """
        mean = self._stats[self.MEAN]
        n_abs = self._stats[self.N_ABS]
        sum_log10_abs = self._stats[self.SUM_LOG10_ABS]

        c = np.full_like(mean, np.nan, dtype=np.float64)

        # primary estimate from magnitude logs
        has_mag = n_abs > 0
        c[has_mag] = 10.0 ** (sum_log10_abs[has_mag] / n_abs[has_mag].astype(np.float64))

        # fallbacks where we have no |x|>0 observations
        none_mag = ~has_mag
        if np.any(none_mag):
            if use_std_fallback:
                sigma = self.stddev(unbiased=unbiased_std)
                fallback = np.maximum(eps, alpha * sigma)
            else:
                fallback = np.full_like(mean, max(eps, alpha * 1.0), dtype=np.float64)
            c[none_mag] = fallback[none_mag]

        # ensure strictly positive
        c = np.maximum(c, eps)
        return c
