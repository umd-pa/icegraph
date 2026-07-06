# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

import numpy as np

from icegraph.common.transforms import TransformSpace
from icegraph.statistics.variants import NANCount

from _helpers import (
    MERGE_A,
    MERGE_B,
    RAW,
    assert_merge_matches_recompute,
    make_bundle,
    ref_nan_count,
)

LINEAR = TransformSpace.LINEAR


# compute

def test_compute_linear_matches_reference():
    stat = NANCount()
    stat.compute(RAW)
    np.testing.assert_allclose(stat.value(LINEAR), ref_nan_count(RAW))


def test_compute_known_values():
    stat = NANCount()
    stat.compute(RAW)
    # col1 carries one NaN, col4 is all NaN
    np.testing.assert_allclose(stat.value(LINEAR), [0.0, 1.0, 0.0, 0.0, 3.0])


def test_compute_does_not_count_inf():
    stat = NANCount()
    stat.compute(np.array([[np.nan], [np.inf], [-np.inf], [1.0]]))
    np.testing.assert_allclose(stat.value(LINEAR), [1.0])


# merge

def test_merge_matches_recompute():
    assert_merge_matches_recompute(NANCount, MERGE_A, MERGE_B)


def test_merge_is_sum_of_counts():
    merged = NANCount.merge(make_bundle(MERGE_A), make_bundle(MERGE_B))
    expected = ref_nan_count(MERGE_A) + ref_nan_count(MERGE_B)
    np.testing.assert_allclose(merged.value(LINEAR), expected)


def test_merge_is_symmetric():
    a, b = make_bundle(MERGE_A), make_bundle(MERGE_B)
    np.testing.assert_allclose(
        NANCount.merge(a, b).value(LINEAR),
        NANCount.merge(b, a).value(LINEAR),
    )
