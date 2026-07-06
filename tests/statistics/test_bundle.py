# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

import numpy as np
import pytest

from icegraph.common.transforms import TransformSpace
from icegraph.statistics import variants
from icegraph.statistics.bundle import StatisticBundle
from icegraph.statistics.variants import Mean, Minimum

# every concrete variant class
ALL_STATS = [getattr(variants, name) for name in variants.__all__]

A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
B = np.array([[2.0, 1.0, 4.0], [3.0, 3.0, 3.0], [5.0, 2.0, 8.0], [1.0, 1.0, 1.0]])


def build_bundle(array) -> StatisticBundle:
    stats = {cls.name: cls() for cls in ALL_STATS}
    return StatisticBundle(stats).compute(np.asarray(array, dtype=float))


def assert_bundles_equal(got: StatisticBundle, want: StatisticBundle) -> None:
    assert got.stats.keys() == want.stats.keys()
    for kind, stat in got.stats.items():
        other = want.get(kind)
        for space in type(stat).spaces:
            np.testing.assert_allclose(
                stat.value(space),
                other.value(space),
                equal_nan=True,
                err_msg=f"{kind} differs in space {space.value}",
            )


# get

def test_get_returns_registered_stat():
    bundle = build_bundle(A)
    stat = bundle.get("mean")
    assert isinstance(stat, Mean)
    assert stat is bundle.stats["mean"]


def test_get_unknown_kind_raises():
    bundle = build_bundle(A)
    with pytest.raises(KeyError):
        bundle.get("does_not_exist")


# compute

def test_compute_returns_self_and_populates_every_stat():
    bundle = StatisticBundle({cls.name: cls() for cls in ALL_STATS})
    result = bundle.compute(A)
    assert result is bundle
    for kind, stat in bundle.stats.items():
        # value() would raise if the stat had not been computed
        stat.value(TransformSpace.LINEAR)


# num_columns

def test_num_columns_matches_input_width():
    assert build_bundle(A).num_columns() == A.shape[1]


def test_num_columns_empty_bundle_raises():
    with pytest.raises(RuntimeError):
        StatisticBundle({}).num_columns()


def test_num_columns_inconsistent_widths_raises():
    bundle = StatisticBundle({"mean": Mean(), "min": Minimum()})
    bundle.stats["mean"].compute(np.ones((3, 4)))
    bundle.stats["min"].compute(np.ones((3, 2)))
    with pytest.raises(RuntimeError):
        bundle.num_columns()


# merge

def test_merge_matches_recompute_on_concatenation():
    merged = StatisticBundle.merge(build_bundle(A), build_bundle(B))
    reference = build_bundle(np.vstack([A, B]))
    assert_bundles_equal(merged, reference)


def test_merge_returns_new_bundle_with_same_keys():
    a, b = build_bundle(A), build_bundle(B)
    merged = StatisticBundle.merge(a, b)
    assert isinstance(merged, StatisticBundle)
    assert merged is not a and merged is not b
    assert merged.stats.keys() == a.stats.keys()


def test_merge_mismatched_stat_sets_raises():
    a = StatisticBundle({"mean": Mean()})
    b = StatisticBundle({"min": Minimum()})
    with pytest.raises(ValueError):
        StatisticBundle.merge(a, b)


# align_to

def test_align_to_reorders_and_returns_self():
    indices = np.array([2, 0, 1])
    bundle = build_bundle(A)
    result = bundle.align_to(indices)
    assert result is bundle
    # aligning after compute equals computing on the column-permuted input
    assert_bundles_equal(bundle, build_bundle(A[:, indices]))


def test_align_to_wrong_length_raises():
    bundle = build_bundle(A)  # 3 columns
    with pytest.raises(ValueError):
        bundle.align_to(np.array([0, 1]))


def test_align_to_non_permutation_raises():
    bundle = build_bundle(A)  # 3 columns
    with pytest.raises(ValueError):
        bundle.align_to(np.array([0, 1, 1]))


# filter_to

def test_filter_to_keeps_masked_columns_and_returns_self():
    mask = np.array([True, False, True])
    bundle = build_bundle(A)
    result = bundle.filter_to(mask)
    assert result is bundle
    assert bundle.num_columns() == 2
    # filtering after compute equals computing on the masked input
    assert_bundles_equal(bundle, build_bundle(A[:, mask]))


def test_filter_to_length_mismatch_raises():
    bundle = build_bundle(A)  # 3 columns
    with pytest.raises(ValueError):
        bundle.filter_to(np.array([True, False]))


# to_struct / from_struct

def test_to_struct_shape():
    struct = build_bundle(A).to_struct()
    assert set(struct) == {"stats"}
    assert set(struct["stats"]) == {cls.name for cls in ALL_STATS}


def test_struct_round_trip():
    bundle = build_bundle(A)
    rebuilt = StatisticBundle.from_struct(bundle.to_struct())
    assert_bundles_equal(rebuilt, bundle)


def test_from_struct_missing_stats_key_raises():
    with pytest.raises(KeyError):
        StatisticBundle.from_struct({})


def test_from_struct_stats_not_dict_raises():
    with pytest.raises(TypeError):
        StatisticBundle.from_struct({"stats": [1, 2, 3]})


def test_from_struct_inconsistent_widths_raises():
    struct = build_bundle(A).to_struct()
    # give one stat a differently sized array than the rest
    struct["stats"]["mean"] = {
        space: np.zeros(A.shape[1] + 1) for space in struct["stats"]["mean"]
    }
    with pytest.raises(ValueError):
        StatisticBundle.from_struct(struct)
