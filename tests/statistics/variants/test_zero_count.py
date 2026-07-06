# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

import numpy as np

from icegraph.common.transforms import TransformSpace
from icegraph.statistics.variants import ZeroCount

from _helpers import (
    MERGE_A,
    MERGE_B,
    RAW,
    assert_merge_matches_recompute,
    make_bundle,
    ref_zero_count,
)

LINEAR = TransformSpace.LINEAR


# compute

def test_compute_linear_matches_reference():
    stat = ZeroCount()
    stat.compute(RAW)
    np.testing.assert_allclose(stat.value(LINEAR), ref_zero_count(RAW))


def test_compute_known_values():
    stat = ZeroCount()
    stat.compute(RAW)
    # col2 has one zero, col3 has two zeros
    np.testing.assert_allclose(stat.value(LINEAR), [0.0, 0.0, 1.0, 2.0, 0.0])


def test_compute_only_counts_exact_zero():
    stat = ZeroCount()
    stat.compute(np.array([[0.0], [-0.0], [1e-12], [np.nan], [0.0]]))
    # +0.0 and -0.0 both count, tiny nonzero and NaN do not
    np.testing.assert_allclose(stat.value(LINEAR), [3.0])


# merge

def test_merge_matches_recompute():
    assert_merge_matches_recompute(ZeroCount, MERGE_A, MERGE_B)


def test_merge_is_sum_of_counts():
    merged = ZeroCount.merge(make_bundle(MERGE_A), make_bundle(MERGE_B))
    expected = ref_zero_count(MERGE_A) + ref_zero_count(MERGE_B)
    np.testing.assert_allclose(merged.value(LINEAR), expected)


def test_merge_is_symmetric():
    a, b = make_bundle(MERGE_A), make_bundle(MERGE_B)
    np.testing.assert_allclose(
        ZeroCount.merge(a, b).value(LINEAR),
        ZeroCount.merge(b, a).value(LINEAR),
    )
