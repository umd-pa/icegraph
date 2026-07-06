# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

import numpy as np

from icegraph.common.transforms import TransformSpace
from icegraph.statistics.variants import PositiveCount

from _helpers import (
    MERGE_A,
    MERGE_B,
    RAW,
    assert_merge_matches_recompute,
    make_bundle,
    ref_positive_count,
)

LINEAR = TransformSpace.LINEAR


# compute

def test_compute_linear_matches_reference():
    stat = PositiveCount()
    stat.compute(RAW)
    np.testing.assert_allclose(stat.value(LINEAR), ref_positive_count(RAW))


def test_compute_known_values():
    stat = PositiveCount()
    stat.compute(RAW)
    # strictly positive finite values per column
    np.testing.assert_allclose(stat.value(LINEAR), [3.0, 2.0, 1.0, 1.0, 0.0])


def test_compute_excludes_zero_negative_and_nonfinite():
    stat = PositiveCount()
    stat.compute(np.array([[0.0], [-1.0], [np.nan], [np.inf], [5.0]]))
    np.testing.assert_allclose(stat.value(LINEAR), [1.0])


# merge

def test_merge_matches_recompute():
    assert_merge_matches_recompute(PositiveCount, MERGE_A, MERGE_B)


def test_merge_is_sum_of_counts():
    merged = PositiveCount.merge(make_bundle(MERGE_A), make_bundle(MERGE_B))
    expected = ref_positive_count(MERGE_A) + ref_positive_count(MERGE_B)
    np.testing.assert_allclose(merged.value(LINEAR), expected)


def test_merge_is_symmetric():
    a, b = make_bundle(MERGE_A), make_bundle(MERGE_B)
    np.testing.assert_allclose(
        PositiveCount.merge(a, b).value(LINEAR),
        PositiveCount.merge(b, a).value(LINEAR),
    )
