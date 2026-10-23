# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from icegraph.data.extractor import ExtractorFactory


ID_COLS = ["Run", "Event", "SubEvent", "SubEventStream"]

# the index columns the table writer books with every table
_INDEX = [("Run", "<u4"), ("Event", "<u4"), ("SubEvent", "<i4"), ("SubEventStream", "<i4"), ("exists", "u1")]


def _table(rows: list[tuple], *fields: tuple[str, str]) -> np.ndarray:
    return np.array(rows, dtype=[*_INDEX, *fields])


def _output(path: Path, tables: dict[str, np.ndarray]) -> str:
    """A file laid out as the table writer writes one."""
    with h5py.File(path, "w") as f:
        for key, table in tables.items():
            f.create_dataset(key, data=table)

    return str(path)


def _extractor(tmp_path: Path, **config):
    gcd = tmp_path / "gcd.i3.zst"
    gcd.touch()

    return ExtractorFactory.create("i3", gcd_path=gcd, ml_suite={}, **config)


def test_events_are_read_off_the_header(tmp_path: Path) -> None:
    extractor = _extractor(tmp_path, include=["features"])
    path = _output(tmp_path / "out.h5", {
        "I3EventHeader": _table([(7, 1, 0, 0, 1, 0.5), (7, 2, 0, 0, 1, 0.5)], ("time_start_mjd", "<f8")),
        "features": _table([(7, 1, 0, 0, 1, 1.0), (7, 1, 0, 0, 1, 2.0), (7, 2, 0, 0, 1, 3.0)], ("charge", "<f8")),
    })

    read = extractor._read(tmp_path / "file.i3.zst", path)
    assert read is not None
    tables, events = read

    assert events.columns == ID_COLS
    assert events["Event"].to_list() == [1, 2]
    assert list(tables) == ["features"]
    assert tables["features"]["charge"].to_list() == [1.0, 2.0, 3.0]


def test_a_file_without_events_keeps_the_tables_it_has(tmp_path: Path) -> None:
    extractor = _extractor(tmp_path, include=["features", "I3CorsikaInfo"])

    # only the S-frames were written
    path = _output(tmp_path / "out.h5", {"I3CorsikaInfo": _table([(0, 0, 0, 0, 1, 1000)], ("n_events", "<i4"))})

    read = extractor._read(tmp_path / "file.i3.zst", path)
    assert read is not None
    tables, events = read

    assert events.is_empty() and events.columns == ID_COLS
    assert list(tables) == ["I3CorsikaInfo"]


def test_a_file_with_events_but_without_a_key_is_refused_or_skipped(tmp_path: Path) -> None:
    path = _output(tmp_path / "out.h5", {"I3EventHeader": _table([(7, 1, 0, 0, 1)])})

    with pytest.raises(KeyError, match="Missing key 'features'"):
        _extractor(tmp_path, include=["features"])._read(tmp_path / "file.i3.zst", path)

    assert _extractor(tmp_path, include=["features"], skip_missing=True)._read(tmp_path / "file.i3.zst", path) is None


def test_the_ids_can_be_configured(tmp_path: Path) -> None:
    path = _output(tmp_path / "out.h5", {"I3EventHeader": _table([(7, 1, 0, 0, 1)])})

    read = _extractor(tmp_path, include=[], ids=["Run", "Event"])._read(tmp_path / "file.i3.zst", path)
    assert read is not None and read[1].columns == ["Run", "Event"]
