# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

import pickle

import numpy as np
import pytest

from icegraph.types.transforms import TransformSpace
from icegraph.statistics import StatisticService


@pytest.fixture
def columns() -> list[str]:
    return ["a", "b", "c"]


@pytest.fixture
def kinds() -> list[str]:
    # minimal set needed for derived statistics + a few basics
    return ["min", "max", "mean", "m2", "finite_count", "positive_count"]


@pytest.fixture
def service(kinds: list[str], columns: list[str]) -> StatisticService:
    return StatisticService(kinds, columns)


@pytest.fixture
def array() -> np.ndarray:
    # shape (N, F) with positives, zeros, negatives, and NaNs
    return np.array(
        [
            [1.0, 10.0, np.nan],
            [2.0, 0.0, 3.0],
            [4.0, -5.0, 6.0],
            [np.nan, 2.0, 12.0],
        ],
        dtype=np.float32,
    )


def test_merge_validations(columns: list[str], kinds: list[str], array: np.ndarray) -> None:
    s = StatisticService(kinds, columns)

    with pytest.raises(TypeError):
        StatisticService.merge("not-an-iterable")  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        StatisticService.merge([])

    with pytest.raises(TypeError):
        StatisticService.merge([s, object()])  # type: ignore[list-item]

    # merge([s]) returns s.copy(), which requires the service be computed
    s.compute_from_array(array)

    out = StatisticService.merge([s])
    assert out is not s
    assert out.columns == s.columns

    # sanity: copied stats should match
    np.testing.assert_allclose(
        out.get("mean", space=TransformSpace.LINEAR),
        s.get("mean", space=TransformSpace.LINEAR),
        equal_nan=True,
    )


def test_sum_works(columns: list[str], kinds: list[str], array: np.ndarray) -> None:
    s1 = StatisticService(kinds, columns)
    s2 = StatisticService(kinds, columns)

    # merging requires underlying stats to exist
    s1.compute_from_array(array)
    s2.compute_from_array(array)

    merged = sum([s1, s2])
    assert isinstance(merged, StatisticService)
    assert merged.columns == columns

    # sum(..., 0) path should also work
    merged2 = sum([s1, s2], 0)
    assert isinstance(merged2, StatisticService)


def test_pickle_roundtrip(service: StatisticService, array: np.ndarray) -> None:
    service.compute_from_array(array)

    blob = pickle.dumps(service)
    restored = pickle.loads(blob)

    assert isinstance(restored, StatisticService)
    assert restored.columns == service.columns

    # spot check that at least one stat matches after round-trip
    np.testing.assert_allclose(
        restored.get("mean", space=TransformSpace.LINEAR),
        service.get("mean", space=TransformSpace.LINEAR),
        equal_nan=True,
    )


def test_get_base_scaling_for_log_and_asinh(service: StatisticService, array: np.ndarray) -> None:
    service.compute_from_array(array)

    # For a statistic with degree=1 (typical for mean/min/max), base conversion should behave like:
    # value(base=b2) == value(base=b1) * log(b1)/log(b2)
    # This relies on Statistic.degree being implemented; if mean.degree != 1, use the exponent.
    mean_base10 = service.get("mean", space=TransformSpace.LOG, base=10)
    mean_base2 = service.get("mean", space=TransformSpace.LOG, base=2)

    # Convert base-10 value to base-2 expectation using the rule above.
    # mean_base(b) = raw_log_value / (log(b) ** degree)
    # so mean_base2 / mean_base10 = (log(10)^d) / (log(2)^d)
    d = service.bundle.get("mean").degree  # depends on internal API; adjust if needed
    factor = (np.log(10.0) ** d) / (np.log(2.0) ** d)
    np.testing.assert_allclose(mean_base2, mean_base10 * factor, rtol=1e-6, atol=0)

    with pytest.raises(ValueError):
        service.get("mean", space=TransformSpace.LOG, base=1)

    with pytest.raises(ValueError):
        service.get("mean", space=TransformSpace.ASINH, base=0)


def test_valid_count_dispatch(service: StatisticService, array: np.ndarray) -> None:
    service.compute_from_array(array)

    finite = service.get("finite_count", space=TransformSpace.LINEAR)
    pos = service.get("positive_count", space=TransformSpace.LINEAR)

    np.testing.assert_allclose(service.valid_count(TransformSpace.LINEAR), finite)
    np.testing.assert_allclose(service.valid_count(TransformSpace.ASINH), finite)
    np.testing.assert_allclose(service.valid_count(TransformSpace.LOG), pos)

    with pytest.raises(TypeError):
        service.valid_count("linear")  # type: ignore[arg-type]


def test_geometric_mean_matches_definition(service: StatisticService, array: np.ndarray) -> None:
    service.compute_from_array(array)

    base = 10
    mean_log = service.get("mean", space=TransformSpace.LOG, base=base)
    geo = service.geometric_mean(base=base)

    np.testing.assert_allclose(geo, base ** mean_log, rtol=1e-6, atol=0)


def test_variance_std_sem_cv_rms_snr_shapes_and_nan_rules(service: StatisticService, array: np.ndarray) -> None:
    service.compute_from_array(array)

    # Basic shape checks
    for fn in (
        lambda: service.variance(),
        lambda: service.std(),
        lambda: service.sem(),
        lambda: service.cv(),
        lambda: service.rms(),
        lambda: service.snr(),
        lambda: service.range(),
    ):
        out = fn()
        assert isinstance(out, np.ndarray)
        assert out.shape == (array.shape[1],)

    # Explicit algebra sanity: std == sqrt(variance)
    var = service.variance(space=TransformSpace.LINEAR, biased=False)
    std = service.std(space=TransformSpace.LINEAR, biased=False)
    np.testing.assert_allclose(std, np.sqrt(var), equal_nan=True)

    # SEM = std / sqrt(n_valid)
    n = service.valid_count(TransformSpace.LINEAR)
    sem = service.sem(space=TransformSpace.LINEAR, biased=False)
    np.testing.assert_allclose(sem, np.where(n > 0, std / np.sqrt(n), np.nan), equal_nan=True)

    # CV = std / mean where mean != 0
    mean = service.get("mean", space=TransformSpace.LINEAR)
    cv = service.cv(space=TransformSpace.LINEAR, biased=False)
    np.testing.assert_allclose(cv, np.where(mean != 0, std / mean, np.nan), equal_nan=True)

    # RMS = sqrt(variance + mean^2)
    rms = service.rms(space=TransformSpace.LINEAR, biased=False)
    np.testing.assert_allclose(rms, np.sqrt(var + np.square(mean)), equal_nan=True)

    # SNR = mean / std where std != 0
    snr = service.snr(space=TransformSpace.LINEAR, biased=False)
    np.testing.assert_allclose(snr, np.where(std != 0, mean / std, np.nan), equal_nan=True)


def test_align_and_filter_and_index_of(service: StatisticService, array: np.ndarray) -> None:
    service.compute_from_array(array)

    # index_of
    assert service.index_of("a") == 0
    assert service.index_of(["a", "c"]) == [0, 2]

    # align_to: reorder columns (must be same set)
    reordered = ["c", "a", "b"]
    service2 = service.copy().align_to(reordered)
    assert service2.columns == reordered
    np.testing.assert_allclose(
        service2.get("mean", space=TransformSpace.LINEAR),
        service.get("mean", space=TransformSpace.LINEAR)[[2, 0, 1]],
        equal_nan=True,
    )

    # filter_to: subset without reordering
    subset = ["a", "c"]
    service3 = service.copy().filter_to(subset)
    assert service3.columns == subset
    np.testing.assert_allclose(
        service3.get("mean", space=TransformSpace.LINEAR),
        service.get("mean", space=TransformSpace.LINEAR)[[0, 2]],
        equal_nan=True,
    )

def test_linear_stats_match_numpy(columns, kinds):
    x = np.array(
        [
            [1.0,  np.nan, 3.0],
            [2.0,  0.0,    5.0],
            [4.0,  -1.0,   np.nan],
        ],
        dtype=np.float64,
    )

    s = StatisticService(kinds, columns)
    s.compute_from_array(x)

    # per-column finite mask
    finite = np.isfinite(x)
    n = finite.sum(axis=0).astype(np.float64)

    # numpy references ignoring NaNs
    ref_mean = np.nanmean(x, axis=0)
    ref_min  = np.nanmin(x, axis=0)
    ref_max  = np.nanmax(x, axis=0)

    np.testing.assert_allclose(s.get("finite_count"), n, rtol=0, atol=0)
    np.testing.assert_allclose(s.get("mean"), ref_mean, equal_nan=True)
    np.testing.assert_allclose(s.get("min"), ref_min, equal_nan=True)
    np.testing.assert_allclose(s.get("max"), ref_max, equal_nan=True)

    # variance / m2 reference
    # m2 = sum (xi - mean)^2 over finite samples
    centered = np.where(finite, x - ref_mean, 0.0)
    ref_m2 = np.sum(centered * centered, axis=0)

    np.testing.assert_allclose(s.get("m2"), ref_m2, rtol=1e-6, atol=0)

    # unbiased variance = m2 / (n-1) when n>1 else nan
    ref_var = np.where(n > 1, ref_m2 / (n - 1), np.nan)
    np.testing.assert_allclose(s.variance(biased=False), ref_var, equal_nan=True)

    # biased variance = m2 / n when n>0 else nan
    ref_var_b = np.where(n > 0, ref_m2 / n, np.nan)
    np.testing.assert_allclose(s.variance(biased=True), ref_var_b, equal_nan=True)

def test_log_space_positive_only_and_base(columns, kinds):
    x = np.array(
        [
            [1.0,  10.0,  0.0],
            [2.0,  -5.0,  4.0],
            [np.nan, 1.0, 8.0],
        ],
        dtype=np.float64,
    )

    s = StatisticService(kinds, columns)
    s.compute_from_array(x)

    pos = np.isfinite(x) & (x > 0)
    npos = pos.sum(axis=0).astype(np.float64)

    # log10 reference mean over positive values only
    log10 = np.where(pos, np.log10(x), np.nan)
    ref_mean_log10 = np.nanmean(log10, axis=0)

    np.testing.assert_allclose(s.get("positive_count"), npos, rtol=0, atol=0)
    np.testing.assert_allclose(
        s.get("mean", space=TransformSpace.LOG, base=10),
        ref_mean_log10,
        equal_nan=True,
    )

    # base conversion check (log2 mean = log10 mean * log10(2) / log10(10) == log10 mean / log10(2))
    ref_mean_log2 = np.nanmean(np.where(pos, np.log2(x), np.nan), axis=0)
    np.testing.assert_allclose(
        s.get("mean", space=TransformSpace.LOG, base=2),
        ref_mean_log2,
        equal_nan=True,
    )

def test_chunked_merge_equals_full_compute(columns, kinds):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(1000, len(columns))).astype(np.float64)

    full = StatisticService(kinds, columns)
    full.compute_from_array(x)

    chunks = np.array_split(x, 5)
    services = []
    for c in chunks:
        s = StatisticService(kinds, columns)
        s.compute_from_array(c)
        services.append(s)

    merged: StatisticService = sum(services)

    for kind in ["mean", "m2", "min", "max"]:
        np.testing.assert_allclose(
            merged.get(kind),
            full.get(kind),
            rtol=1e-6,
            equal_nan=True,
        )
