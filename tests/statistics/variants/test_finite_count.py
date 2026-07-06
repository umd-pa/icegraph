# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

import numpy as np

from icegraph.common.transforms import TransformSpace
from icegraph.statistics.variants import FiniteCount

from _helpers import (
    MERGE_A,
    MERGE_B,
    RAW,
    assert_merge_matches_recompute,
    make_bundle,
    ref_finite_count,
)

LINEAR = TransformSpace.LINEAR


# compute

def test_compute_linear_matches_reference():
    stat = FiniteCount()
    stat.compute(RAW)
    np.testing.assert_allclose(stat.value(LINEAR), ref_finite_count(RAW))


def test_compute_known_values():
    stat = FiniteCount()
    stat.compute(RAW)
    # col1 has one NaN, col4 is all NaN
    np.testing.assert_allclose(stat.value(LINEAR), [3.0, 2.0, 3.0, 3.0, 0.0])


def test_compute_excludes_nan_and_inf():
    stat = FiniteCount()
    stat.compute(np.array([[1.0], [np.nan], [np.inf], [-np.inf], [2.0]]))
    np.testing.assert_allclose(stat.value(LINEAR), [2.0])


# merge

def test_merge_matches_recompute():
    assert_merge_matches_recompute(FiniteCount, MERGE_A, MERGE_B)


def test_merge_is_sum_of_counts():
    merged = FiniteCount.merge(make_bundle(MERGE_A), make_bundle(MERGE_B))
    expected = ref_finite_count(MERGE_A) + ref_finite_count(MERGE_B)
    np.testing.assert_allclose(merged.value(LINEAR), expected)


def test_merge_is_symmetric():
    a, b = make_bundle(MERGE_A), make_bundle(MERGE_B)
    np.testing.assert_allclose(
        FiniteCount.merge(a, b).value(LINEAR),
        FiniteCount.merge(b, a).value(LINEAR),
    )
