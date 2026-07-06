# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

import numpy as np
import pytest

from icegraph.common.transforms import TransformSpace
from icegraph.statistics.variants import Mean

from _helpers import (
    MERGE_A,
    MERGE_B,
    RAW,
    as2d,
    assert_merge_matches_recompute,
    make_bundle,
    ref_mean,
    transform,
)

LINEAR = TransformSpace.LINEAR

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


# compute

def test_compute_linear_known_values():
    stat = Mean()
    stat.compute(RAW)
    # col4 is all-NaN, mean is NaN
    np.testing.assert_allclose(
        stat.value(LINEAR), [7.0 / 3.0, 4.0, 2.0, 7.0 / 3.0, np.nan], equal_nan=True
    )


@pytest.mark.parametrize("space", list(Mean.spaces), ids=lambda s: s.value)
def test_compute_matches_reference_each_space(space):
    stat = Mean()
    stat.compute(RAW)
    np.testing.assert_allclose(
        stat.value(space), ref_mean(transform(space, RAW)), equal_nan=True
    )


def test_compute_ignores_nan():
    stat = Mean()
    stat.compute(np.array([[2.0], [np.nan], [4.0]]))
    np.testing.assert_allclose(stat.value(LINEAR), [3.0])


def test_compute_log_uses_positive_values_only():
    # log space is defined over strictly positive values, negatives/zeros drop out
    stat = Mean()
    stat.compute(np.array([[1.0], [-5.0], [0.0], [np.e]]))
    # mean of log(1)=0 and log(e)=1 is 0.5
    np.testing.assert_allclose(stat.value(TransformSpace.LOG), [0.5])


# merge

def test_merge_matches_recompute():
    assert_merge_matches_recompute(Mean, MERGE_A, MERGE_B)


def test_merge_linear_equals_pooled_mean():
    merged = Mean.merge(make_bundle(MERGE_A), make_bundle(MERGE_B))
    expected = ref_mean(np.vstack([as2d(MERGE_A), as2d(MERGE_B)]))
    np.testing.assert_allclose(merged.value(LINEAR), expected, equal_nan=True)


def test_merge_is_symmetric():
    a, b = make_bundle(MERGE_A), make_bundle(MERGE_B)
    for space in Mean.spaces:
        np.testing.assert_allclose(
            Mean.merge(a, b).value(space),
            Mean.merge(b, a).value(space),
            equal_nan=True,
        )


def test_merge_all_empty_column_is_nan():
    # a column that is non-positive everywhere has no log-space samples
    a = np.array([[-1.0], [-2.0]])
    b = np.array([[-3.0], [-4.0]])
    merged = Mean.merge(make_bundle(a), make_bundle(b))
    assert np.isnan(merged.value(TransformSpace.LOG)).all()
