# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

import pickle

import numpy as np
import pytest

from icegraph.common.transforms import TransformSpace
from icegraph.statistics import variants
from icegraph.statistics.service import StatisticService

LINEAR = TransformSpace.LINEAR
LOG = TransformSpace.LOG
ASINH = TransformSpace.ASINH

# every registered kind, so all derived statistics are available
KINDS = [getattr(variants, name).name for name in variants.__all__]

# strictly positive single-column data with hand-checkable statistics
DATA = np.array([1.0, 2.0, 4.0])
# a second column with a negative so finite_count != positive_count
SIGNED = np.array([-1.0, 2.0, 4.0])


def build_service(array, kinds=KINDS) -> StatisticService:
    service = StatisticService(kinds)
    service.compute_from_array(np.asarray(array, dtype=float).reshape(-1, 1))
    return service


# construction

def test_init_builds_requested_kinds():
    service = StatisticService(["mean", "min", "max"])
    assert set(service.bundle.stats) == {"mean", "min", "max"}
    assert isinstance(service.bundle.stats["mean"], variants.Mean)


def test_init_unknown_kind_raises():
    with pytest.raises(Exception):
        StatisticService(["not_a_real_stat"])


# get: raw access and log/asinh base scaling

def test_get_linear_returns_raw_value():
    service = build_service(DATA)
    np.testing.assert_allclose(service.get("mean"), [DATA.mean()])


def test_get_linear_ignores_base():
    service = build_service(DATA)
    # base is only meaningful for LOG/ASINH, so it must be ignored here
    np.testing.assert_allclose(service.get("mean", space=LINEAR, base=0), [DATA.mean()])


def test_get_log_rescales_to_base():
    service = build_service(DATA)
    # mean is degree 1: natural-log value divided by ln(base) == log_base mean
    np.testing.assert_allclose(service.get("mean", space=LOG, base=10), [np.mean(np.log10(DATA))])


def test_get_asinh_rescales_by_degree():
    service = build_service(DATA)
    expected = np.mean(np.arcsinh(DATA)) / np.log(10)
    np.testing.assert_allclose(service.get("mean", space=ASINH, base=10), [expected])


@pytest.mark.parametrize("base", [0, -2, 1])
def test_get_invalid_base_for_nonlinear_raises(base):
    service = build_service(DATA)
    with pytest.raises(ValueError):
        service.get("mean", space=LOG, base=base)


# valid_count

def test_valid_count_linear_and_asinh_use_finite_count():
    service = build_service(SIGNED)  # finite_count = 3
    np.testing.assert_allclose(service.valid_count(LINEAR), [3.0])
    np.testing.assert_allclose(service.valid_count(ASINH), [3.0])


def test_valid_count_log_uses_positive_count():
    service = build_service(SIGNED)  # positive_count = 2
    np.testing.assert_allclose(service.valid_count(LOG), [2.0])


def test_valid_count_invalid_space_raises():
    service = build_service(DATA)
    with pytest.raises(TypeError):
        service.valid_count("linear")  # not a TransformSpace


# derived statistics (linear space cross-checked against numpy)

def test_variance_unbiased_and_biased():
    service = build_service(DATA)
    np.testing.assert_allclose(service.variance(), [DATA.var(ddof=1)])
    np.testing.assert_allclose(service.variance(biased=True), [DATA.var(ddof=0)])


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_variance_insufficient_count_is_nan():
    # a single sample -> unbiased denominator (n-1) is 0 -> NaN
    service = build_service(np.array([5.0]))
    assert np.isnan(service.variance()).all()


def test_std_matches_sqrt_variance():
    service = build_service(DATA)
    np.testing.assert_allclose(service.std(), [DATA.std(ddof=1)])


def test_range_is_max_minus_min():
    service = build_service(DATA)
    np.testing.assert_allclose(service.range(), [DATA.max() - DATA.min()])


def test_sem_matches_std_over_sqrt_n():
    service = build_service(DATA)
    np.testing.assert_allclose(service.sem(), [DATA.std(ddof=1) / np.sqrt(DATA.size)])


def test_cv_matches_std_over_mean():
    service = build_service(DATA)
    np.testing.assert_allclose(service.cv(), [DATA.std(ddof=1) / DATA.mean()])


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_cv_zero_mean_is_nan():
    service = build_service(np.array([-2.0, 0.0, 2.0]))  # mean == 0
    assert np.isnan(service.cv()).all()


def test_rms_biased_is_root_mean_square():
    service = build_service(DATA)
    # biased variance + mean^2 == E[x^2]
    np.testing.assert_allclose(service.rms(biased=True), [np.sqrt(np.mean(DATA**2))])


def test_rms_unbiased_matches_formula():
    service = build_service(DATA)
    expected = np.sqrt(DATA.var(ddof=1) + DATA.mean() ** 2)
    np.testing.assert_allclose(service.rms(), [expected])


def test_snr_matches_mean_over_std():
    service = build_service(DATA)
    np.testing.assert_allclose(service.snr(), [DATA.mean() / DATA.std(ddof=1)])


def test_geometric_mean_is_base_independent():
    service = build_service(DATA)
    # geometric mean of (1, 2, 4) == 8 ** (1/3) == 2, regardless of base
    np.testing.assert_allclose(service.geometric_mean(base=10), [2.0])
    np.testing.assert_allclose(service.geometric_mean(base=2), [2.0])


def test_variance_log_space_rescales_correctly():
    service = build_service(DATA)
    np.testing.assert_allclose(
        service.variance(space=LOG, base=10), [np.var(np.log10(DATA), ddof=1)]
    )


# merge

def test_merge_matches_recompute_on_concatenation():
    a, b = build_service(DATA), build_service(SIGNED)
    merged = StatisticService.merge([a, b])
    reference = build_service(np.concatenate([DATA, SIGNED]))
    for kind in ("mean", "min", "max"):
        np.testing.assert_allclose(merged.get(kind), reference.get(kind), equal_nan=True)


def test_merge_single_returns_independent_copy():
    a = build_service(DATA)
    merged = StatisticService.merge([a])
    assert merged is not a
    np.testing.assert_allclose(merged.get("mean"), a.get("mean"))


def test_merge_empty_raises():
    with pytest.raises(ValueError):
        StatisticService.merge([])


@pytest.mark.parametrize("bad", ["ab", b"ab", 5, [True, 0]])
def test_merge_type_check(bad):
    with pytest.raises(TypeError):
        StatisticService.merge(bad)


# addition operators

def test_add_zero_returns_self():
    a = build_service(DATA)
    assert (a + 0) is a
    assert (0 + a) is a


def test_add_two_services_merges():
    a, b = build_service(DATA), build_service(SIGNED)
    reference = build_service(np.concatenate([DATA, SIGNED]))
    np.testing.assert_allclose((a + b).get("mean"), reference.get("mean"))


def test_sum_over_services_uses_radd():
    a, b = build_service(DATA), build_service(SIGNED)
    reference = build_service(np.concatenate([DATA, SIGNED]))
    total = sum([a, b])  # sum seeds with 0 -> exercises __radd__
    assert isinstance(total, StatisticService)
    np.testing.assert_allclose(total.get("mean"), reference.get("mean"))


# serialization and copy

def test_struct_round_trip():
    service = build_service(DATA)
    rebuilt = StatisticService.from_struct(service.to_struct())
    for kind in KINDS:
        np.testing.assert_allclose(rebuilt.get(kind), service.get(kind), equal_nan=True)


def test_pickle_round_trip():
    service = build_service(DATA)
    rebuilt = pickle.loads(pickle.dumps(service))
    assert isinstance(rebuilt, StatisticService)
    np.testing.assert_allclose(rebuilt.get("mean"), service.get("mean"))
    np.testing.assert_allclose(rebuilt.variance(), service.variance())


def test_copy_is_independent_deep_copy():
    service = build_service(DATA)
    clone = service.copy()
    assert clone is not service
    assert clone.bundle is not service.bundle
    # mutating the clone's underlying data must not touch the original
    clone.bundle.stats["mean"]._values[LINEAR][:] = -999.0
    np.testing.assert_allclose(service.get("mean"), [DATA.mean()])
