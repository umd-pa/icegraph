# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

import numpy as np
import pytest

from icegraph.common.transforms import TransformSpace
from icegraph.statistics import variants

from icegraph.statistics.statistic import Statistic


def _concrete_statistics() -> list[type[Statistic]]:
    return [getattr(variants, name) for name in variants.__all__]

STATS = _concrete_statistics()

if not STATS:
    pytest.skip(
        "No concrete Statistic subclasses found; set STATISTICS_PACKAGE correctly.",
        allow_module_level=True,
    )


# fixtures

@pytest.fixture(params=STATS, ids=lambda c: c.__name__)
def stat_cls(request) -> type[Statistic]:
    return request.param


@pytest.fixture
def stat(stat_cls) -> Statistic:
    return stat_cls()


def _fill_random(instance: Statistic, n_cols: int = 4, lead: tuple[int, ...] = ()) -> Statistic:
    """Populate _values for all tracked spaces with random arrays of equal width."""
    rng = np.random.default_rng(0)
    for space in type(instance).spaces:
        instance._values[space] = rng.standard_normal((*lead, n_cols))
    return instance


def _fill_known(instance: Statistic, n_cols: int = 4, lead: tuple[int, ...] = ()) -> np.ndarray:
    """Populate _values where each column equals its index (for reorder checks)."""
    base = np.arange(n_cols, dtype=float)
    arr = np.broadcast_to(base, (*lead, n_cols)).copy()
    for space in type(instance).spaces:
        instance._values[space] = arr.copy()
    return base


# construction

def test_constructs_with_no_args(stat_cls):
    inst = stat_cls()
    assert isinstance(inst, Statistic)
    assert inst._values == {}


def test_class_invariants(stat_cls):
    assert isinstance(stat_cls.name, str)
    assert isinstance(stat_cls.degree, int) and stat_cls.degree >= 0
    assert isinstance(stat_cls.spaces, tuple) and len(stat_cls.spaces) >= 1
    assert all(isinstance(s, TransformSpace) for s in stat_cls.spaces)
    assert len(set(stat_cls.spaces)) == len(stat_cls.spaces)


def test_every_tracked_space_has_a_transform(stat_cls):
    for space in stat_cls.spaces:
        assert space in Statistic.transform
        assert callable(Statistic.transform[space])


# tests for value()

def test_value_returns_stored_array(stat):
    _fill_random(stat)
    for space in type(stat).spaces:
        assert stat.value(space) is stat._values[space]


def test_value_unsupported_space_raises(stat, stat_cls):
    untracked = [s for s in TransformSpace if s not in stat_cls.spaces]
    if not untracked:
        pytest.skip("statistic tracks every space")
    with pytest.raises(KeyError):
        stat.value(untracked[0])


def test_value_before_compute_raises(stat, stat_cls):
    # space is tracked but nothing computed yet should yield missing key
    with pytest.raises(KeyError):
        stat.value(stat_cls.spaces[0])


# tests for num_columns()

def test_num_columns_before_values_raises(stat):
    with pytest.raises(RuntimeError):
        stat.num_columns()


def test_num_columns_returns_width(stat):
    _fill_random(stat, n_cols=7)
    assert stat.num_columns() == 7


def test_num_columns_ignores_leading_dims(stat):
    _fill_random(stat, n_cols=5, lead=(2, 3))
    assert stat.num_columns() == 5


def test_num_columns_inconsistent_widths_raises(stat, stat_cls):
    if len(stat_cls.spaces) < 2:
        pytest.skip("needs >= 2 spaces to be inconsistent")
    _fill_random(stat, n_cols=4)
    stat._values[stat_cls.spaces[0]] = np.zeros((3,))  # different width
    with pytest.raises(RuntimeError):
        stat.num_columns()


# tests for align()

def test_align_to_reorders_columns(stat):
    base = _fill_known(stat, n_cols=4)
    indices = np.array([3, 2, 1, 0])
    stat.align_to(indices)
    for space in type(stat).spaces:
        np.testing.assert_array_equal(stat._values[space], base[indices])


def test_align_to_preserves_leading_dims(stat):
    base = _fill_known(stat, n_cols=4, lead=(5,))
    indices = np.array([1, 0, 3, 2])
    stat.align_to(indices)
    for space in type(stat).spaces:
        arr = stat._values[space]
        assert arr.shape == (5, 4)
        np.testing.assert_array_equal(arr, np.broadcast_to(base[indices], (5, 4)))


def test_align_to_non_1d_raises(stat):
    _fill_random(stat, n_cols=4)
    with pytest.raises(ValueError):
        stat.align_to(np.zeros((2, 2), dtype=int))


def test_align_to_length_mismatch_raises(stat):
    _fill_random(stat, n_cols=4)
    with pytest.raises(ValueError):
        stat.align_to(np.array([0, 1, 2]))  # 3 != 4


def test_align_to_out_of_bounds_raises(stat):
    _fill_random(stat, n_cols=4)
    with pytest.raises(IndexError):
        stat.align_to(np.array([0, 1, 2, 4]))


def test_align_to_duplicates_raises(stat):
    _fill_random(stat, n_cols=4)
    with pytest.raises(ValueError):
        stat.align_to(np.array([0, 1, 2, 2]))


# tests for filter_to()

def test_filter_to_keeps_masked_columns(stat):
    base = _fill_known(stat, n_cols=4)
    mask = np.array([True, False, True, False])
    stat.filter_to(mask)
    for space in type(stat).spaces:
        np.testing.assert_array_equal(stat._values[space], base[mask])


def test_filter_to_non_1d_raises(stat):
    _fill_random(stat, n_cols=4)
    with pytest.raises(ValueError):
        stat.filter_to(np.ones((2, 2), dtype=bool))


def test_filter_to_length_mismatch_raises(stat):
    _fill_random(stat, n_cols=4)
    with pytest.raises(ValueError):
        stat.filter_to(np.array([True, False]))  # 2 != 4


# tests for to_struct() and from_struct()

def test_to_struct_keys(stat, stat_cls):
    _fill_random(stat)
    struct = stat.to_struct()
    assert set(struct) == {s.value for s in stat_cls.spaces}


def test_struct_round_trip(stat, stat_cls):
    _fill_random(stat)
    struct = stat.to_struct()
    rebuilt = stat_cls.from_struct(struct)
    assert set(rebuilt._values) == set(stat._values)
    for space in stat_cls.spaces:
        np.testing.assert_array_equal(rebuilt._values[space], stat._values[space])


def test_from_struct_missing_space_raises(stat, stat_cls):
    _fill_random(stat)
    struct = stat.to_struct()
    struct.pop(stat_cls.spaces[0].value)  # drop a required space
    with pytest.raises(ValueError):
        stat_cls.from_struct(struct)


def test_from_struct_unsupported_space_raises(stat, stat_cls):
    untracked = [s for s in TransformSpace if s not in stat_cls.spaces]
    if not untracked:
        pytest.skip("statistic tracks every space")
    _fill_random(stat)
    struct = stat.to_struct()
    struct[untracked[0].value] = np.zeros((4,))
    with pytest.raises(ValueError):
        stat_cls.from_struct(struct)


# tests for compute()

def test_compute_empty_raises(stat):
    with pytest.raises(ValueError):
        stat.compute(np.array([]))


def test_compute_3d_raises(stat):
    with pytest.raises(ValueError):
        stat.compute(np.zeros((2, 2, 2)))

def test_compute_rejects_complex_input(stat):
    with pytest.raises(ValueError):
        stat.compute(np.array([1 + 2j, 3 + 0j]))


# tests for __init_subclass__() checks

def _make_stat(**ns):
    """Build a concrete Statistic subclass with overridable class vars."""
    body = {
        "_compute": lambda self, array: array,
        "_merge": classmethod(lambda cls, a, b, space: np.array([])),
        "degree": 1,
        "spaces": (TransformSpace.LINEAR,),
    }
    body.update(ns)
    return lambda: type("TmpStat", (Statistic,), body)


def test_subclass_valid_definition_ok():
    _make_stat()()  # should not raise


@pytest.mark.parametrize(
    "overrides, exc",
    [
        ({"degree": object()}, TypeError),               # degree not int
        ({"degree": 1.5}, TypeError),                    # degree not int
        ({"degree": -1}, ValueError),                    # degree negative
        ({"spaces": [TransformSpace.LINEAR]}, TypeError),  # spaces not tuple
        ({"spaces": ()}, ValueError),                    # spaces empty
        ({"spaces": ("linear",)}, TypeError),            # wrong element type
        ({"spaces": (TransformSpace.LINEAR, TransformSpace.LINEAR)}, ValueError),  # dupes
    ],
)
def test_subclass_invalid_definition_raises(overrides, exc):
    with pytest.raises(exc):
        _make_stat(**overrides)()