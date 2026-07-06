# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

import warnings

import numpy as np

from icegraph.common.transforms import TransformSpace
from icegraph.statistics import variants
from icegraph.statistics.bundle import StatisticBundle
from icegraph.statistics.transforms import (
    linear_transform,
    log_transform,
    asinh_transform,
)

# every concrete variant class
ALL_STATS = [getattr(variants, name) for name in variants.__all__]

RAW = np.array(
    [
        [1.0, 3.0, -2.0, 0.0, np.nan],
        [2.0, np.nan, 8.0, 0.0, np.nan],
        [4.0, 5.0, 0.0, 7.0, np.nan],
    ]
)

# two partitions used to check the merge invariant (same column count)
MERGE_A = np.array(
    [
        [1.0, 3.0, -2.0, 0.0, np.nan],
        [2.0, np.nan, 8.0, 0.0, np.nan],
        [4.0, 5.0, 0.0, 7.0, np.nan],
    ]
)
MERGE_B = np.array(
    [
        [-1.0, 2.0, 9.0, 0.0, 4.0],
        [6.0, np.nan, -3.0, 0.0, 3.0],
        [0.0, 1.0, 4.0, np.nan, 2.0],
        [2.0, 2.0, 2.0, 5.0, 9.0],
    ]
)

_TRANSFORM = {
    TransformSpace.LINEAR: linear_transform,
    TransformSpace.LOG: log_transform,
    TransformSpace.ASINH: asinh_transform,
}


def as2d(array) -> np.ndarray:
    """Coerce input to a float 2D array the way ``Statistic.compute`` does."""
    a = np.asarray(array, dtype=float)
    return a.reshape(-1, 1) if a.ndim == 1 else a


def transform(space: TransformSpace, array) -> np.ndarray:
    """Apply a space transform to a raw array (returns a 2D array)."""
    return _TRANSFORM[space](as2d(array))


def make_bundle(array) -> StatisticBundle:
    """Build a bundle holding every variant, computed on ``array``."""
    stats = {cls.name: cls() for cls in ALL_STATS}
    bundle = StatisticBundle(stats)
    # all-NaN columns in the sample data make nanmean/nanmin warn; that path is
    # exercised intentionally, so silence the expected warning here.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        bundle.compute(as2d(array))
    return bundle


# reference reductions (independent of the implementations under test)

def _columns(array) -> list[np.ndarray]:
    a = as2d(array)
    return [a[:, j] for j in range(a.shape[1])]


def ref_total_count(raw) -> np.ndarray:
    a = as2d(raw)
    return np.full(a.shape[1], float(a.shape[0]))


def ref_finite_count(raw) -> np.ndarray:
    return np.array([float(np.sum(np.isfinite(c))) for c in _columns(raw)])


def ref_nan_count(raw) -> np.ndarray:
    return np.array([float(np.sum(np.isnan(c))) for c in _columns(raw)])


def ref_positive_count(raw) -> np.ndarray:
    return np.array(
        [float(np.sum(np.isfinite(c) & (c > 0.0))) for c in _columns(raw)]
    )


def ref_zero_count(raw) -> np.ndarray:
    return np.array(
        [float(np.sum(np.isfinite(c) & (c == 0.0))) for c in _columns(raw)]
    )


def _drop_nan(c: np.ndarray) -> np.ndarray:
    return c[~np.isnan(c)]


def ref_min(transformed) -> np.ndarray:
    out = []
    for c in _columns(transformed):
        vals = _drop_nan(c)
        out.append(float(vals.min()) if vals.size else np.nan)
    return np.array(out, dtype=float)


def ref_max(transformed) -> np.ndarray:
    out = []
    for c in _columns(transformed):
        vals = _drop_nan(c)
        out.append(float(vals.max()) if vals.size else np.nan)
    return np.array(out, dtype=float)


def ref_mean(transformed) -> np.ndarray:
    out = []
    for c in _columns(transformed):
        vals = _drop_nan(c)
        out.append(float(vals.sum() / vals.size) if vals.size else np.nan)
    return np.array(out, dtype=float)


def ref_m2(transformed) -> np.ndarray:
    out = []
    for c in _columns(transformed):
        vals = _drop_nan(c)
        if vals.size == 0:
            out.append(0.0)  # matches np.nansum over an all-nan column
            continue
        mean = vals.sum() / vals.size
        out.append(float(np.sum((vals - mean) ** 2)))
    return np.array(out, dtype=float)


def assert_merge_matches_recompute(cls, a_raw, b_raw) -> None:
    """merge(compute(A), compute(B)) must equal compute(concat(A, B))."""
    a = make_bundle(a_raw)
    b = make_bundle(b_raw)

    merged = cls.merge(a, b)
    reference = make_bundle(np.vstack([as2d(a_raw), as2d(b_raw)])).get(cls.name)

    for space in cls.spaces:
        np.testing.assert_allclose(
            merged.value(space),
            reference.value(space),
            equal_nan=True,
            err_msg=f"{cls.__name__} merge mismatch in space {space.value}",
        )
