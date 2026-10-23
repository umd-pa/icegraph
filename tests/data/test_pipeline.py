# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import polars as pl
import pytest

from icegraph.common.data import AttributeDomain
from icegraph.data import Pipeline
from icegraph.data.collector import Collector, CollectorConfig
from icegraph.data.envelope import Envelope
from icegraph.data.extractor import Extractor
from icegraph.data.processor import ProcessorFactory
from icegraph.data.quiver import QuiverIPC
from icegraph.data.writer import WriterFactory
from icegraph.engine.services.record.reader import ReaderFactory, ReaderContext


ID_COLS = ["Run", "Event"]

# define a generic extractor
class TableExtractor(Extractor[dict[str, Any]]):
    """Reads one table per input file, each row an event. A file without rows holds no events."""
    name: ClassVar[str] = "test-tables"
    version: ClassVar[int] = 1

    file_ext: ClassVar[str] = ".parquet"

    @classmethod
    def validate_config(cls, config: dict[str, Any]) -> dict[str, Any]:
        return config

    def build(self) -> None:
        return

    def extract(self, item: Path) -> Envelope | None:
        rows = pl.read_parquet(item)
        root = tempfile.mkdtemp(dir=self._ctx.scratch)

        env = Envelope(
            quiver=QuiverIPC.from_data({"events": rows} if rows.height else {}, root),
            events=rows.select(ID_COLS)
        )

        return env


def _inputs(root: Path, sizes: dict[str, list[int]]) -> pl.DataFrame:
    """One file per size in each directory, 0 for a file without events. Returns every event written."""
    rng = np.random.default_rng(0)
    written = []

    run = 0
    for directory, counts in sizes.items():
        (root / directory).mkdir(parents=True)

        for i, n in enumerate(counts):
            run += 1
            rows = pl.DataFrame(
                {"Run": np.full(n, run, np.uint32), "Event": np.arange(n, dtype=np.uint32), "x": rng.normal(size=n)}
            )
            rows.write_parquet(root / directory / f"file{i}.parquet")
            written.append(rows)

    return pl.concat(written)


def _run(source: list[Path], outdir: Path, **collector: Any) -> None:
    pipeline = Pipeline(
        source, outdir,
        extractor=TableExtractor({}),
        processors=[
            ProcessorFactory.create("select", key="events"),
            ProcessorFactory.create("commit", cols=["x"]),  # keyed on the event ids by default
        ],
        writer=WriterFactory.create("zarr", chunk_size=1, prefix="test"),
        collector=Collector(CollectorConfig(**collector))
    )

    with pipeline:
        pipeline.execute(nproc=3)


def _shards(outdir: Path) -> list[tuple[dict[str, Any], pl.DataFrame]]:
    """The local attributes and rows of every shard written."""
    shards = []
    for path in sorted(outdir.iterdir()):
        reader = ReaderFactory.create("zarr")
        reader.attach(ReaderContext(path=path))

        block = reader.read(np.arange(len(reader)))
        rows = pl.DataFrame({name: block.columns[name].values for name in [*ID_COLS, "x"]})

        shards.append((reader.attrs[AttributeDomain.LOCAL], rows))
        reader.close()

    return shards


def test_every_file_is_counted_in_one_shard(tmp_path: Path) -> None:
    sizes = {"a": [0, 5, 3, 0, 0, 8, 2], "b": [4, 0, 6, 0]}
    events = _inputs(tmp_path / "in", sizes)
    outdir = tmp_path / "out"

    # small enough that a shard holds a few files
    _run([tmp_path / "in" / "a", tmp_path / "in" / "b"], outdir, min_bytes=2000)

    shards = _shards(outdir)
    assert [path.name for path in sorted(outdir.iterdir())] == [f"test.shard.{i:06d}.zarr" for i in range(len(shards))]

    sources = [Path(origin) for local, _ in shards for origin in local["sources"]]
    assert sorted(sources) == sorted((tmp_path / "in" / d / f"file{i}.parquet") for d, n in sizes.items() for i in range(len(n)))

    for local, rows in shards:
        # a shard never mixes directories, and always holds an event
        assert len({Path(origin).parent for origin in local["sources"]}) == 1
        assert rows.height > 0

    written = pl.concat([rows for _, rows in shards]).sort(ID_COLS)
    assert written.equals(events.sort(ID_COLS))


def test_files_without_events_in_a_directory_of_their_own(tmp_path: Path) -> None:
    sizes = {"a": [3, 4], "b": [0, 0]}
    _inputs(tmp_path / "in", sizes)
    source = [tmp_path / "in" / "a", tmp_path / "in" / "b"]

    # they cannot be counted toward their own directory
    with pytest.raises(RuntimeError, match="No file of group"):
        _run(source, tmp_path / "out")

    # unless every directory is one set
    _run(source, tmp_path / "all", group_by="all")
    assert sum(len(local["sources"]) for local, _ in _shards(tmp_path / "all")) == 4


def test_a_file_given_twice_is_refused(tmp_path: Path) -> None:
    _inputs(tmp_path / "in", {"a": [3]})

    with pytest.raises(ValueError, match="more than once"):
        _run([tmp_path / "in" / "a", tmp_path / "in" / "a" / "file0.parquet"], tmp_path / "out")


def test_a_non_empty_output_directory_warns(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    _inputs(tmp_path / "in", {"a": [3]})
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "old.zarr").mkdir()

    with caplog.at_level(logging.WARNING):
        _run([tmp_path / "in" / "a"], tmp_path / "out")

    assert "is not empty" in caplog.text
