# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from icegraph.data.collector import Collector, CollectorConfig
from icegraph.data.envelope import Envelope
from icegraph.data.quiver import QuiverIPC


ID_COLS = ["Run", "Event"]


def _file(tmp_path: Path, name: str, events: list[int], *, directory: str = "a", run: int = 1) -> Envelope:
    """The envelope an extractor emits for one file, holding the events given by number."""
    ids = pl.DataFrame({"Run": [run] * len(events), "Event": events}, schema={"Run": pl.UInt32, "Event": pl.UInt32})

    # a file without events still has its S-frames
    tables = {"info": pl.DataFrame({"n": [len(events)]})}
    if events:
        tables["events"] = ids.with_columns(x=pl.col("Event").cast(pl.Float64))

    env = Envelope(
        quiver=QuiverIPC.from_data(tables, tmp_path / "scratch" / directory / name),
        events=ids if events else pl.DataFrame(schema=ID_COLS)
    )
    env.set_local_attr("origin", str(tmp_path / directory / f"{name}.i3.zst"))
    env.set_local_attr("dropped", {"coincident": len(events)})
    env.set_global_attr("gcd", "gcd.i3.zst")
    env.state["extractor"]["src_file_ext"] = "i3.zst"

    return env


def _collect(files: list[Envelope], **config) -> list[Envelope]:
    return list(Collector(CollectorConfig(**config)).collect(files))


def _names(block: Envelope) -> list[str]:
    return [Path(origin).name.removesuffix(".i3.zst") for origin in block.get_local_attr("sources")]


def test_blocks_hold_whole_files_and_close_once_full(tmp_path: Path) -> None:
    files = [_file(tmp_path, f"f{i}", list(range(10 * i, 10 * i + 10))) for i in range(5)]

    # every file holds the same number of rows, so two fill a block
    blocks = _collect(files, min_bytes=2 * files[0].quiver.nbytes)

    assert [_names(block) for block in blocks] == [["f0", "f1"], ["f2", "f3"], ["f4"]]
    assert [block.shard for block in blocks] == [0, 1, 2]


def test_files_without_events_join_the_open_block(tmp_path: Path) -> None:
    files = [
        _file(tmp_path, "e0", []),
        _file(tmp_path, "f1", [1, 2]),
        _file(tmp_path, "e2", []),
        _file(tmp_path, "e3", []),
        _file(tmp_path, "f4", [4]),
        _file(tmp_path, "e5", []),
    ]

    # with no minimum every file with events starts a block, but one without never does
    blocks = _collect(files)

    assert [_names(block) for block in blocks] == [["e0", "f1", "e2", "e3"], ["f4", "e5"]]
    assert [block.events.height for block in blocks] == [2, 1]


def test_a_block_is_one_envelope_over_its_files(tmp_path: Path) -> None:
    files = [_file(tmp_path, "f0", [1, 2]), _file(tmp_path, "e1", []), _file(tmp_path, "f2", [3])]
    [block] = _collect(files, min_bytes=10 ** 9)

    # tables are stacked in file order, a key held by only some files reads as theirs
    assert block.quiver["events"]["Event"].to_list() == [1, 2, 3]
    assert block.quiver["info"]["n"].to_list() == [2, 0, 1]
    assert [("events" in part, "info" in part) for part in block.quiver.parts] == [(True, True), (False, True), (True, True)]

    assert block.events.equals(pl.concat([files[0].events, files[2].events]))
    assert block.ids == ID_COLS

    # each file keeps its own local attributes, without the origin keying them
    sources = block.get_local_attr("sources")
    assert list(sources) == [file.get_local_attr("origin") for file in files]
    assert [source["dropped"]["coincident"] for source in sources.values()] == [2, 0, 1]
    assert block.get_local_attr("origin") is None

    assert block.get_global_attr("gcd") == "gcd.i3.zst"
    assert block.state["extractor"]["src_file_ext"] == "i3.zst"


def test_group_by_parent_keeps_directories_apart(tmp_path: Path) -> None:
    files = [
        _file(tmp_path, "a0", [1], directory="a"),
        _file(tmp_path, "b0", [1], directory="b", run=2),
        _file(tmp_path, "a1", [2], directory="a"),
        _file(tmp_path, "b1", [], directory="b"),
    ]

    by_parent = _collect(files, min_bytes=10 ** 9)
    assert sorted(_names(block) for block in by_parent) == [["a0", "a1"], ["b0", "b1"]]

    [everything] = _collect(files, min_bytes=10 ** 9, group_by="all")
    assert _names(everything) == ["a0", "b0", "a1", "b1"]


def test_a_group_without_events_raises(tmp_path: Path) -> None:
    files = [_file(tmp_path, "a0", [1], directory="a"), _file(tmp_path, "b0", [], directory="b")]

    with pytest.raises(RuntimeError, match="No file of group .* holds events"):
        _collect(files)

    # collected with every other file, it is counted in their block
    [block] = _collect(files, group_by="all")
    assert _names(block) == ["a0", "b0"]


def test_repeated_event_ids_raise_naming_the_files(tmp_path: Path) -> None:
    files = [_file(tmp_path, "f0", [1, 2]), _file(tmp_path, "e1", []), _file(tmp_path, "f2", [2, 3])]

    with pytest.raises(RuntimeError, match=r"'Event': 2.*f0\.i3\.zst.*f2\.i3\.zst"):
        _collect(files, min_bytes=10 ** 9)

    # files in separate blocks never meet
    assert len(_collect(files)) == 2


def test_files_must_share_global_attributes(tmp_path: Path) -> None:
    files = [_file(tmp_path, "f0", [1]), _file(tmp_path, "f1", [2])]
    files[1].set_global_attr("gcd", "other.i3.zst")

    with pytest.raises(RuntimeError, match="different global attributes"):
        _collect(files, min_bytes=10 ** 9)


def test_files_must_identify_events_by_the_same_columns(tmp_path: Path) -> None:
    files = [_file(tmp_path, "f0", [1]), _file(tmp_path, "f1", [2])]
    files[1].events = files[1].events.rename({"Event": "Id"})

    with pytest.raises(RuntimeError, match="sets ids to"):
        _collect(files, min_bytes=10 ** 9)


def test_a_file_without_event_ids_raises(tmp_path: Path) -> None:
    file = _file(tmp_path, "f0", [1])
    file.events = pl.DataFrame()

    with pytest.raises(RuntimeError, match="set no event ids"):
        _collect([file])


def test_a_file_collected_twice_raises(tmp_path: Path) -> None:
    files = [_file(tmp_path, "f0", [1]), _file(tmp_path, "f0", [2], run=2)]

    with pytest.raises(RuntimeError, match="collected more than once"):
        _collect(files, min_bytes=10 ** 9)


def test_ids_resolve_to_the_event_columns(tmp_path: Path) -> None:
    env = _file(tmp_path, "f0", [1])
    env.tmp["events"] = env.quiver["events"]
    env.active = "events"

    assert env.resolve_cols("__ids__") == ID_COLS
    assert env.resolve_cols(["__ids__", "x"]) == [*ID_COLS, "x"]

    with pytest.raises(RuntimeError, match="no event ids"):
        Envelope(quiver=env.quiver).resolve_cols("__ids__")
