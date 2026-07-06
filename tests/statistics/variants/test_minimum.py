# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

import numpy as np
import pytest

from icegraph.common.transforms import TransformSpace
from icegraph.statistics.variants import Minimum

from _helpers import (
    MERGE_A,
    MERGE_B,
    RAW,
    assert_merge_matches_recompute,
    make_bundle,
    ref_min,
    transform,
)

LINEAR = TransformSpace.LINEAR

# col4 is all-NaN, nanmin over an empty slice returns NaN (and warns)
pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


# compute

def test_compute_linear_known_values():
    stat = Minimum()
    stat.compute(RAW)
    np.testing.assert_allclose(
        stat.value(LINEAR), [1.0, 3.0, -2.0, 0.0, np.nan], equal_nan=True
    )


@pytest.mark.parametrize("space", list(Minimum.spaces), ids=lambda s: s.value)
def test_compute_matches_reference_each_space(space):
    stat = Minimum()
    stat.compute(RAW)
    np.testing.assert_allclose(
        stat.value(space), ref_min(transform(space, RAW)), equal_nan=True
    )


def test_compute_ignores_nan():
    stat = Minimum()
    stat.compute(np.array([[np.nan], [3.0], [-1.0], [np.nan]]))
    np.testing.assert_allclose(stat.value(LINEAR), [-1.0])


# merge

def test_merge_matches_recompute():
    assert_merge_matches_recompute(Minimum, MERGE_A, MERGE_B)


def test_merge_is_elementwise_min():
    merged = Minimum.merge(make_bundle(MERGE_A), make_bundle(MERGE_B))
    expected = np.fmin(ref_min(MERGE_A), ref_min(MERGE_B))
    np.testing.assert_allclose(merged.value(LINEAR), expected)


def test_merge_is_symmetric():
    a, b = make_bundle(MERGE_A), make_bundle(MERGE_B)
    for space in Minimum.spaces:
        np.testing.assert_allclose(
            Minimum.merge(a, b).value(space),
            Minimum.merge(b, a).value(space),
            equal_nan=True,
        )
