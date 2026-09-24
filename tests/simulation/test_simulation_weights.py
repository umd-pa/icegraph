# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

import shutil
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterator, NamedTuple

import numpy as np
import polars as pl
import pytest

simweights = pytest.importorskip("simweights")

from icegraph.data.envelope import Envelope
from icegraph.data.quiver import QuiverIPC
from icegraph.data.processor import ProcessorFactory
from icegraph.data.types import StageContext
from icegraph.data.writer.factory import WriterFactory
from icegraph.engine.services import ServiceManager
from icegraph.engine.services.decode.records.variants.standard.surface import build_surfaces


Tables = dict[str, dict[str, np.ndarray]]

ID_COLS = ["Run", "Event", "SubEvent", "SubEventStream"]

# the stored key the generation columns are packed under, as in the processing configs
KEY = "generation"

_FLUX_CONFIGS: dict[str, dict[str, Any]] = {
    "nugen": {"source": "nuflux", "name": "stub"},
    "corsika": {"source": "simweights", "name": "GaisserH4a"},
}

# the per event table of each simulation, which the weight columns are keyed on
_EVENT_TABLE: dict[str, str] = {
    "nugen": "I3MCWeightDict",
    "corsika": "PolyplopiaPrimary",
}

# pdgid to mass number
_CORSIKA_SPECIES: dict[int, int] = {
    2212: 1,
    1000020040: 4,
    1000070140: 14,
    1000130270: 27,
    1000260560: 56,
}

# showers thrown per file of each primary
_CORSIKA_THROWN: dict[int, int] = {
    2212: 8_000,
    1000020040: 4_000,
    1000070140: 2_000,
    1000130270: 1_000,
    1000260560: 1_000,
}


class _StubNuFlux:
    """Stands in for a nuflux model, which simweights calls as getFlux(pdgid, energy, cos_zen).

    Depends on every argument, so a column swapped, dropped or mis-cast on the way
    through storage changes the weights.
    """

    def getFlux(self, pdgid: np.ndarray, energy: np.ndarray, cos_zen: np.ndarray) -> np.ndarray:  # noqa: N802
        pdgid = np.asarray(pdgid, dtype=np.float64)
        flavor = np.where(pdgid > 0, 1.0, 0.8) * np.abs(pdgid) / 12.0

        return 1e-18 * flavor * np.asarray(energy) ** -2.7 * (1.5 + np.asarray(cos_zen))


### SIMULATED FILES

def _ids(run: int, n: int) -> dict[str, np.ndarray]:
    return {
        "Run":              np.full(n, run, dtype=np.uint32),
        "Event":            np.arange(n, dtype=np.uint32),
        "SubEvent":         np.zeros(n, dtype=np.uint32),
        "SubEventStream":   np.zeros(n, dtype=np.uint32),
    }


def _features(ids: dict[str, np.ndarray], rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Pulse level rows for the file's events, several per event and in no particular order."""
    n = len(ids["Run"])
    index = np.repeat(np.arange(n), rng.integers(1, 4, n))
    rng.shuffle(index)

    return {**{name: column[index] for name, column in ids.items()}, "charge": rng.uniform(0.0, 10.0, len(index))}


def _nugen_file(
        rng: np.random.Generator, run: int, n: int, *, spatial: str, types: tuple[int, ...],
        min_log: float = 2.0, max_log: float = 7.0
) -> Tables:
    """One file of a nugen set. I3MCWeightDict is a map of doubles, so every column is float64."""
    table = {
        **_ids(run, n),
        "PrimaryNeutrinoType":      rng.choice(types, n).astype(np.float64),
        "PrimaryNeutrinoEnergy":    10 ** rng.uniform(min_log, max_log, n),
        "PrimaryNeutrinoZenith":    np.arccos(rng.uniform(-1.0, 1.0, n)),
        "TotalWeight":              rng.uniform(0.1, 1.0, n),
        "MinZenith":                np.full(n, 0.0),
        "MaxZenith":                np.full(n, np.pi),
        "MinEnergyLog":             np.full(n, min_log),
        "MaxEnergyLog":             np.full(n, max_log),
        "PowerLawIndex":            np.full(n, 2.0),
        "NEvents":                  np.full(n, 10_000.0),
        "TypeWeight":               np.full(n, 0.5),
    }

    # simweights picks the injection geometry from which of these the table carries
    match spatial:
        case "circle":
            table["InjectionSurfaceR"] = np.full(n, 1200.0)
        case "cylinder":
            table["CylinderHeight"] = np.full(n, 1600.0)
            table["CylinderRadius"] = np.full(n, 800.0)

    return {"I3MCWeightDict": table, "features": _features(_ids(run, n), rng)}


def _corsika_file(
        rng: np.random.Generator, run: int, n: int, *, kept: tuple[int, ...] = tuple(_CORSIKA_SPECIES),
        emin: float | None = None, emax: float = 1e9
) -> Tables:
    """One file for a 5 component corsika set, each species thrown above its own threshold unless one is given."""
    theta_min, theta_max = 0.0, np.deg2rad(89.99)
    thresholds = {p: 600.0 * a if emin is None else emin for p, a in _CORSIKA_SPECIES.items()}

    pdgid = rng.choice(kept, n)
    low = np.array([thresholds[p] for p in pdgid], dtype=np.float64)

    primary = {
        **_ids(run, n),
        "type":     pdgid.astype(np.float64),
        "energy":   10 ** rng.uniform(np.log10(low), np.log10(emax)),
        "zenith":   np.arccos(rng.uniform(np.cos(theta_max), np.cos(theta_min), n)),
    }

    # one S-frame row per primary thrown, kept or not
    thrown = np.array(list(_CORSIKA_SPECIES), dtype=np.int32)
    rows = len(thrown)

    info = {
        "primary_type":     thrown,
        "n_events":         np.array([_CORSIKA_THROWN[p] for p in thrown], dtype=np.int32),
        "oversampling":     np.ones(rows, dtype=np.int32),
        "cylinder_height":  np.full(rows, 1600.0),
        "cylinder_radius":  np.full(rows, 800.0),
        "min_zenith":       np.full(rows, theta_min),
        "max_zenith":       np.full(rows, theta_max),
        "min_energy":       np.array([thresholds[p] for p in thrown], dtype=np.float64),
        "max_energy":       np.full(rows, emax),
        "power_law_index":  np.full(rows, -2.0),
    }

    return {"I3CorsikaInfo": info, "PolyplopiaPrimary": primary, "features": _features(_ids(run, n), rng)}


def _nugen_dataset(
        rng: np.random.Generator, spatial: str, *,
        flavor: int = 14, first_run: int = 1, last: tuple[int, ...] | None = None,
        min_log: float = 2.0, max_log: float = 7.0
) -> list[Tables]:
    types = (flavor, -flavor)
    energy = {"min_log": min_log, "max_log": max_log}
    return [
        _nugen_file(rng, first_run, 300, spatial=spatial, types=types, **energy),
        _nugen_file(rng, first_run + 1, 250, spatial=spatial, types=types, **energy),
        _nugen_file(rng, first_run + 2, 40, spatial=spatial, types=last or types, **energy),
    ]


def _corsika_dataset(
        rng: np.random.Generator, *,
        first_run: int = 11, last: tuple[int, ...] = tuple(_CORSIKA_SPECIES), emin: float | None = None, emax: float = 1e9
) -> list[Tables]:
    energy = {"emin": emin, "emax": emax}
    return [
        _corsika_file(rng, first_run, 300, **energy),
        _corsika_file(rng, first_run + 1, 250, **energy),
        _corsika_file(rng, first_run + 2, 40, kept=last, **energy),
    ]


# between them these store every generation table the processor knows
DATASETS: dict[str, tuple[str, Callable[[np.random.Generator], list[Tables]]]] = {
    "nugen-circle":     ("nugen", lambda rng: _nugen_dataset(rng, "circle")),
    "nugen-cylinder":   ("nugen", lambda rng: _nugen_dataset(rng, "cylinder")),
    "corsika":          ("corsika", lambda rng: _corsika_dataset(rng)),

    # a file keeps no event of a type it threw, as a small or heavily filtered file will
    "nugen-file-without-nubar":     ("nugen", lambda rng: _nugen_dataset(rng, "cylinder", last=(14,))),
    "corsika-file-without-iron":    ("corsika", lambda rng: _corsika_dataset(rng, last=tuple(_CORSIKA_SPECIES)[:-1])),
}

# two sets of one type whose energy ranges overlap
OVERLAPPING: dict[str, Callable[[np.random.Generator], list[list[Tables]]]] = {
    "nugen": lambda rng: [
        _nugen_dataset(rng, "cylinder", first_run=1, min_log=2.0, max_log=7.0),
        _nugen_dataset(rng, "cylinder", first_run=4, min_log=5.0, max_log=8.0),
    ],
    "corsika": lambda rng: [
        _corsika_dataset(rng, first_run=11),
        _corsika_dataset(rng, first_run=21, emin=1e8, emax=1e10),
    ],
}


### ICEGRAPH

def _write_file(tables: Tables, *, simulation: dict[str, Any] | None, scratch: Path, outdir: Path, origin: str) -> None:
    """Process and write one file with the stages the processing configs run, the simulation ones if given."""
    env = Envelope(quiver=QuiverIPC.from_data({key: pl.DataFrame(table) for key, table in tables.items()}, scratch))
    env.state["alias"]["event_ids"] = ID_COLS
    env.set_local_attr("origin", origin)

    stages = [
        ("select", {"key": "features"}),
        ("compress", {"to": "compressed", "by": "event_ids", "cols": ["charge"], "out": "features",
                      "dtype": "float32", "override_dtypes": True}),
    ]
    if simulation is not None:
        stages += [
            ("i3-simulation", {**simulation, "ids": "event_ids"}),
            ("select", {"key": "generation"}),
            ("compress", {"to": "compressed", "by": "event_ids", "cols": "__all__", "out": KEY, "dtype": "float64"}),
        ]
    stages += [
        ("select", {"key": "compressed"}),
        ("commit", {"ids": "event_ids", "cols": ["features"] if simulation is None else ["features", KEY]}),
    ]

    for name, kwargs in stages:
        env = ProcessorFactory.create(name, **kwargs)._process(env)
        assert env is not None

    # the last stage devivifies before it runs
    env.devivify()

    writer = WriterFactory.create("zarr", chunk_size=8)
    writer.attach(StageContext(src=(), dst=None, scratch=scratch, index=0, total=1, outdir=outdir))
    writer._process(env)


def _write_dataset(files: list[Tables], simulation: str, root: Path, *, sim_code: int | None = None) -> Path:
    outdir = root / "out"
    outdir.mkdir(parents=True)

    config = {"type": simulation} if sim_code is None else {"type": simulation, "sim_code": sim_code}
    for i, tables in enumerate(files):
        _write_file(tables, simulation=config, scratch=root / f"scratch{i}", outdir=outdir, origin=f"file{i}.i3.zst")

    return outdir


@contextmanager
def _services(sources: list[Path], simulations: list[str]) -> Iterator[ServiceManager]:
    """The record and decode services, built as the engine builds them."""
    services = ServiceManager.from_config({
        "record": {"source": sources, "reader": {"name": "zarr", "kwargs": {}}, "cache_size": 4},
        "decode": {
            "attrs": {"name": "standard", "kwargs": {}},
            "records": {"name": "standard", "kwargs": {"flux": {s: _FLUX_CONFIGS[s] for s in simulations}}},
        },
    }, debug=False)

    try:
        yield services
    finally:
        services.close()


def _icegraph_weights(services: ServiceManager, blocks: int = 4) -> pl.DataFrame:
    """The weights the dataset hands the engine, keyed by event, read in blocks that cross shards."""
    record = services.require("record")
    decode = services.require("decode")

    frames = []
    for indices in np.array_split(np.arange(len(record)), blocks):
        block = record.read(indices)
        weights = decode.load_weights(block).numpy()

        assert weights.dtype == np.float32

        frames.append(
            pl.DataFrame({name: block.columns[name].values for name in ID_COLS})
            .with_columns(icegraph=pl.Series(weights))
        )

    return pl.concat(frames)


### SIMWEIGHTS

def _simweights(files: list[Tables], simulation: str) -> Any:
    """simweights over one dataset held as one file."""
    merged = {
        key: {col: np.concatenate([tables[key][col] for tables in files]) for col in files[0][key]}
        for key in files[0] if key != "features"
    }

    # S-frames count the files themselves, and simweights reads them only without nfiles
    if simulation == "corsika":
        return simweights.CorsikaWeighter(merged)

    return simweights.NuGenWeighter(merged, nfiles=len(files))


def _simweights_weights(datasets: list[list[Tables]], simulation: str, flux: Any) -> pl.DataFrame:
    """simweights over datasets of one simulation, combined as simweights combines them."""
    weights = sum(_simweights(files, simulation) for files in datasets).get_weights(flux)

    # an event outside the surface weighs 0 on both sides, which would compare equal
    assert (weights > 0).all()

    events = [tables[_EVENT_TABLE[simulation]] for files in datasets for tables in files]
    ids = {name: np.concatenate([table[name] for table in events]) for name in ID_COLS}

    # the engine sees float32
    return pl.DataFrame(ids).with_columns(simweights=pl.Series(weights.astype(np.float32)))


def _flux(simulation: str, nuflux: _StubNuFlux) -> Any:
    """The model the configured flux builds to."""
    return nuflux if simulation == "nugen" else simweights.GaisserH4a()


def _assert_same_weights(icegraph: pl.DataFrame, expected: pl.DataFrame) -> None:
    joined = expected.join(icegraph, on=ID_COLS, how="inner", validate="1:1")

    # every event is weighted exactly once, and against the right event
    assert joined.height == expected.height == icegraph.height

    np.testing.assert_array_equal(joined["icegraph"].to_numpy(), joined["simweights"].to_numpy())


def _assert_same_surface(rebuilt: Any, expected: Any) -> None:
    assert set(rebuilt.spectra) == set(expected.spectra)

    for pdgid, specs in expected.spectra.items():
        got = rebuilt.spectra[pdgid]

        # get_epdf sums components in list order, so order matters for identical weights
        assert len(got) == len(specs), f"pdgid {pdgid}"

        for g, e in zip(got, specs):
            assert int(g.pdgid) == int(e.pdgid)
            assert g.nevents == e.nevents, f"pdgid {pdgid}: generated events"
            assert [type(d) for d in g.dists] == [type(d) for d in e.dists], f"pdgid {pdgid}"

            for gd, ed in zip(g.dists, e.dists):
                # simweights compares distributions without their column, so check it here
                # vars also covers everything derived from the parameters on construction
                assert gd.columns == ed.columns, f"pdgid {pdgid}: {gd!r}"
                assert vars(gd) == vars(ed), f"pdgid {pdgid}: {gd!r}"

    assert rebuilt == expected


### TESTS

class Written(NamedTuple):
    simulation: str
    files:      list[Tables]
    outdir:     Path


@pytest.fixture(scope="module", params=list(DATASETS))
def written(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Written:
    """A dataset processed and written by icegraph."""
    simulation, make = DATASETS[request.param]
    files = make(np.random.default_rng(0))

    return Written(simulation, files, _write_dataset(files, simulation, tmp_path_factory.mktemp(request.param)))


@pytest.fixture
def nuflux(monkeypatch: pytest.MonkeyPatch) -> _StubNuFlux:
    """Stub nuflux so the decoder's flux is one the test can also hand to simweights."""
    flux = _StubNuFlux()
    monkeypatch.setitem(sys.modules, "nuflux", SimpleNamespace(makeFlux=lambda name: flux))

    return flux


def test_weights_match_simweights_over_the_whole_dataset(written: Written, nuflux: _StubNuFlux) -> None:
    expected = _simweights_weights([written.files], written.simulation, _flux(written.simulation, nuflux))

    with _services([written.outdir], [written.simulation]) as services:
        _assert_same_weights(_icegraph_weights(services), expected)


def test_rebuilt_surface_matches_simweights_over_the_whole_dataset(written: Written) -> None:
    """Where the weights disagree, the surface says which species and why."""
    with _services([written.outdir], [written.simulation]) as services:
        surfaces = build_surfaces(services.require("record").attrs)

    assert surfaces is not None and list(surfaces.surfaces) == [written.simulation]
    _assert_same_surface(surfaces.surfaces[written.simulation], _simweights(written.files, written.simulation).surface)


@pytest.mark.parametrize("dataset", list(DATASETS))
def test_weights_match_simweights_after_dropping_a_file(dataset: str, tmp_path: Path, nuflux: _StubNuFlux) -> None:
    simulation, make = DATASETS[dataset]
    files = make(np.random.default_rng(0))
    outdir = _write_dataset(files, simulation, tmp_path)

    # dropped after processing, leaving the file that kept fewer types with one that kept all
    shutil.rmtree(outdir / "file0.i3.zarr")

    expected = _simweights_weights([files[1:]], simulation, _flux(simulation, nuflux))

    with _services([outdir], [simulation]) as services:
        _assert_same_weights(_icegraph_weights(services), expected)


def test_mixed_datasets_match_simweights_per_dataset(tmp_path: Path, nuflux: _StubNuFlux) -> None:
    rng = np.random.default_rng(1)
    datasets = {
        "nugen": _nugen_dataset(rng, "circle"),
        "corsika": _corsika_dataset(rng),
    }

    # each dataset loaded from its own source, as the training config lists them, one
    # named by its configured code and the other by the one discovered
    sources = [
        _write_dataset(datasets["nugen"], "nugen", tmp_path / "nugen", sim_code=23123),
        _write_dataset(datasets["corsika"], "corsika", tmp_path / "corsika"),
    ]

    # each simulation weighted by simweights on its own, against its own flux
    expected = pl.concat([
        _simweights_weights([files], simulation, _flux(simulation, nuflux))
        for simulation, files in datasets.items()
    ])

    with _services(sources, list(datasets)) as services:
        surfaces = build_surfaces(services.require("record").attrs)
        _assert_same_weights(_icegraph_weights(services), expected)

    assert surfaces is not None and surfaces.codes[23123] == "nugen"


def test_nugen_sets_of_different_flavors_are_weighted_apart(tmp_path: Path, nuflux: _StubNuFlux) -> None:
    rng = np.random.default_rng(3)
    datasets = [
        _nugen_dataset(rng, "cylinder", flavor=14, first_run=1),
        _nugen_dataset(rng, "cylinder", flavor=12, first_run=4),
    ]

    sources = [_write_dataset(files, "nugen", tmp_path / str(i)) for i, files in enumerate(datasets)]
    expected = _simweights_weights(datasets, "nugen", nuflux)

    with _services(sources, ["nugen"]) as services:
        _assert_same_weights(_icegraph_weights(services), expected)


@pytest.mark.parametrize("simulation", list(OVERLAPPING))
def test_sets_with_overlapping_energy_are_weighted_together(simulation: str, tmp_path: Path, nuflux: _StubNuFlux) -> None:
    # simweights combines datasets by summing their weighters, so an event in the overlap
    # is weighted against the generation of both sets
    datasets = OVERLAPPING[simulation](np.random.default_rng(5))

    sources = [_write_dataset(files, simulation, tmp_path / str(i)) for i, files in enumerate(datasets)]
    expected = _simweights_weights(datasets, simulation, _flux(simulation, nuflux))

    with _services(sources, [simulation]) as services:
        surfaces = build_surfaces(services.require("record").attrs)
        _assert_same_weights(_icegraph_weights(services), expected)

    # however the sets were coded, their type is weighted against one surface
    assert surfaces is not None and list(surfaces.surfaces) == [simulation]


def test_nugen_file_of_several_flavours_needs_a_sim_code(tmp_path: Path) -> None:
    tables = _nugen_file(np.random.default_rng(4), 1, 50, spatial="cylinder", types=(14, 12))

    with pytest.raises(RuntimeError, match="Set 'sim_code'"):
        _write_file(tables, simulation={"type": "nugen"}, scratch=tmp_path / "scratch", outdir=tmp_path, origin="file0.i3.zst")


def test_real_data_has_no_weights(tmp_path: Path) -> None:
    outdir = tmp_path / "out"
    outdir.mkdir()

    features = {"features": _features(_ids(1, 50), np.random.default_rng(2))}
    _write_file(features, simulation=None, scratch=tmp_path / "scratch", outdir=outdir, origin="file0.i3.zst")

    with _services([outdir], []) as services:
        record = services.require("record")
        weights = services.require("decode").load_weights(record.read(np.arange(len(record))))

    assert weights.numel() == 0
