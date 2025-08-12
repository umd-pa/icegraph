# tests/test_stats.py
import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

from icegraph.utils import Statistics as Stats


def make_demo_df():
    # a: normal numeric; b: constant; c: NaN/±inf edge cases; d: all-NaN
    return pd.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0],            # mean=2.5, M2=5.0
        "b": [10.0, 10.0, 10.0, 10.0],        # mean=10,  M2=0
        "c": [np.nan, 0.0, np.inf, -np.inf],  # finite-only => [0.0]
        "d": [np.nan, np.nan, np.nan, np.nan] # no finite values
    })


def finite_stats_reference(df: pd.DataFrame):
    # Reference impl (finite-only), mirrors Stats.from_dataframe
    vals = df.to_numpy(dtype=np.float64)
    finite = np.isfinite(vals)

    n = finite.sum(axis=0)
    safe = np.where(finite, vals, np.nan)
    mn = np.where(n > 0, np.nanmin(safe, axis=0), np.inf)
    mx = np.where(n > 0, np.nanmax(safe, axis=0), -np.inf)
    sums = np.where(finite, vals, 0.0).sum(axis=0)
    mean = sums / np.maximum(n, 1)
    diffs = np.where(finite, vals - mean, 0.0)
    M2 = (diffs * diffs).sum(axis=0)
    nan_count = df.isna().sum(axis=0).to_numpy(dtype=np.int64)
    return n, mn, mx, mean, M2, nan_count


def test_from_dataframe_matches_reference():
    df = make_demo_df()
    s = Stats.from_dataframe(df)

    n, mn, mx, mean, M2, nan_count = finite_stats_reference(df)

    assert s.columns == list(df.columns)
    npt.assert_array_equal(s.n, n)
    npt.assert_allclose(s.min, mn)
    npt.assert_allclose(s.max, mx)
    npt.assert_allclose(s.mean, mean)
    npt.assert_allclose(s.M2, M2)
    npt.assert_array_equal(s.nan_count, nan_count)


def test_zero_count_column_behavior():
    df = make_demo_df()
    s = Stats.from_dataframe(df)

    idx_d = s.columns.index("d")
    assert s.n[idx_d] == 0
    assert np.isposinf(s.min[idx_d])        # +inf by design
    assert np.isneginf(s.max[idx_d])        # -inf by design
    assert s.mean[idx_d] == 0.0             # placeholder (ignored in merges)
    assert s.M2[idx_d] == 0.0


def test_merge_equivalent_to_concat():
    df1 = pd.DataFrame({"x": [0.0, 1.0, 2.0], "y": [5.0, 5.0, 5.0]})
    df2 = pd.DataFrame({"x": [3.0, 4.0],       "y": [5.0, 5.0]})

    s1 = Stats.from_dataframe(df1)
    s2 = Stats.from_dataframe(df2)
    merged = s1.merge(s2)

    s_ref = Stats.from_dataframe(pd.concat([df1, df2], ignore_index=True))

    assert merged.columns == s_ref.columns
    npt.assert_array_equal(merged.n, s_ref.n)
    npt.assert_allclose(merged.min, s_ref.min)
    npt.assert_allclose(merged.max, s_ref.max)
    npt.assert_allclose(merged.mean, s_ref.mean)
    npt.assert_allclose(merged.M2, s_ref.M2)
    npt.assert_array_equal(merged.nan_count, s_ref.nan_count)


def test_merge_many_and_append_dataframe():
    dfA = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [0.0, np.nan, 2.0]})
    dfB = pd.DataFrame({"a": [4.0],           "b": [2.0]})
    dfC = pd.DataFrame({"a": [5.0, 6.0],      "b": [np.nan, 4.0]})

    sA = Stats.from_dataframe(dfA)
    sB = Stats.from_dataframe(dfB)
    sC = Stats.from_dataframe(dfC)

    merged_many = Stats.merge_many([sA, sB, sC])
    s_ref = Stats.from_dataframe(pd.concat([dfA, dfB, dfC], ignore_index=True))

    for got in [merged_many]:
        assert got.columns == s_ref.columns
        npt.assert_array_equal(got.n, s_ref.n)
        npt.assert_allclose(got.min, s_ref.min)
        npt.assert_allclose(got.max, s_ref.max)
        npt.assert_allclose(got.mean, s_ref.mean)
        npt.assert_allclose(got.M2, s_ref.M2)


def test_alignment_and_merge_with_reordered_columns():
    df = pd.DataFrame({"u": [1.0, 2.0, 3.0], "v": [9.0, 8.0, 7.0]})
    s1 = Stats.from_dataframe(df)               # columns ['u','v']
    s2 = Stats.from_dataframe(df[["v", "u"]])   # columns ['v','u']

    # Ensure alignment works inside merge
    merged = s1.merge(s2)
    s_ref = Stats.from_dataframe(pd.concat([df, df], ignore_index=True))

    assert merged.columns == ["u", "v"]
    npt.assert_allclose(merged.mean, s_ref.mean)
    npt.assert_allclose(merged.min, s_ref.min)
    npt.assert_allclose(merged.max, s_ref.max)
    npt.assert_allclose(merged.M2, s_ref.M2)

