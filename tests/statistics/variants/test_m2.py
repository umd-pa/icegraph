# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

import numpy as np
import pytest

from icegraph.common.transforms import TransformSpace
from icegraph.statistics.variants import WelfordM2

from _helpers import (
    MERGE_A,
    MERGE_B,
    RAW,
    as2d,
    assert_merge_matches_recompute,
    make_bundle,
    ref_m2,
    transform,
)

LINEAR = TransformSpace.LINEAR

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


# compute

def test_compute_linear_known_values():
    stat = WelfordM2()
    stat.compute(RAW)
    # M2 = sum (x - mean)^2 per column, col4 is all-NaN so nansum yields 0.0
    np.testing.assert_allclose(
        stat.value(LINEAR), [14.0 / 3.0, 2.0, 56.0, 98.0 / 3.0, 0.0]
    )


@pytest.mark.parametrize("space", list(WelfordM2.spaces), ids=lambda s: s.value)
def test_compute_matches_reference_each_space(space):
    stat = WelfordM2()
    stat.compute(RAW)
    np.testing.assert_allclose(
        stat.value(space), ref_m2(transform(space, RAW)), equal_nan=True
    )


def test_compute_equals_variance_times_count():
    stat = WelfordM2()
    stat.compute(RAW)
    t = transform(LINEAR, RAW)
    cols = [j for j in range(t.shape[1]) if np.isfinite(t[:, j]).any()]
    expected = np.array(
        [np.nanvar(t[:, j]) * np.sum(np.isfinite(t[:, j])) for j in cols]
    )
    np.testing.assert_allclose(stat.value(LINEAR)[cols], expected)


def test_compute_ignores_nan():
    stat = WelfordM2()
    stat.compute(np.array([[2.0], [np.nan], [4.0]]))
    # mean 3 -> (2-3)^2 + (4-3)^2 = 2
    np.testing.assert_allclose(stat.value(LINEAR), [2.0])


# merge

def test_merge_matches_recompute():
    assert_merge_matches_recompute(WelfordM2, MERGE_A, MERGE_B)


def test_merge_linear_equals_pooled_m2():
    merged = WelfordM2.merge(make_bundle(MERGE_A), make_bundle(MERGE_B))
    expected = ref_m2(np.vstack([as2d(MERGE_A), as2d(MERGE_B)]))
    np.testing.assert_allclose(merged.value(LINEAR), expected, equal_nan=True)


def test_merge_is_symmetric():
    a, b = make_bundle(MERGE_A), make_bundle(MERGE_B)
    for space in WelfordM2.spaces:
        np.testing.assert_allclose(
            WelfordM2.merge(a, b).value(space),
            WelfordM2.merge(b, a).value(space),
            equal_nan=True,
        )


def test_merge_all_empty_column_is_nan():
    # no log-space samples in either partition, merged M2 is NaN
    a = np.array([[-1.0], [-2.0]])
    b = np.array([[-3.0], [-4.0]])
    merged = WelfordM2.merge(make_bundle(a), make_bundle(b))
    assert np.isnan(merged.value(TransformSpace.LOG)).all()
