# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

import numpy as np

from icegraph.common.transforms import TransformSpace
from icegraph.statistics.variants import TotalCount

from _helpers import (
    MERGE_A,
    MERGE_B,
    RAW,
    assert_merge_matches_recompute,
    make_bundle,
    ref_total_count,
)

LINEAR = TransformSpace.LINEAR


# compute

def test_compute_linear_matches_reference():
    stat = TotalCount()
    stat.compute(RAW)
    np.testing.assert_allclose(stat.value(LINEAR), ref_total_count(RAW))


def test_compute_counts_every_row_including_nan():
    stat = TotalCount()
    stat.compute(RAW)
    # total count is the row count regardless of NaN / sign
    np.testing.assert_allclose(stat.value(LINEAR), [3.0, 3.0, 3.0, 3.0, 3.0])


def test_compute_1d_input_is_single_column():
    stat = TotalCount()
    stat.compute(np.array([1.0, np.nan, 3.0, 4.0]))
    np.testing.assert_allclose(stat.value(LINEAR), [4.0])


# merge

def test_merge_matches_recompute():
    assert_merge_matches_recompute(TotalCount, MERGE_A, MERGE_B)


def test_merge_is_sum_of_row_counts():
    merged = TotalCount.merge(make_bundle(MERGE_A), make_bundle(MERGE_B))
    expected = ref_total_count(MERGE_A) + ref_total_count(MERGE_B)
    np.testing.assert_allclose(merged.value(LINEAR), expected)


def test_merge_is_symmetric():
    a, b = make_bundle(MERGE_A), make_bundle(MERGE_B)
    np.testing.assert_allclose(
        TotalCount.merge(a, b).value(LINEAR),
        TotalCount.merge(b, a).value(LINEAR),
    )
