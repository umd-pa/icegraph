# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from threading import Thread
from typing import Self, Any, cast
from pathlib import Path
import tempfile
import multiprocessing as mp

import yaml
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn, TimeElapsedColumn

from icegraph.common.files import SourceType, Source
from icegraph.ui import console

from .shared.queue import IterableQueue
from .config import Config
from .types import StageContext, Envelope

from .extractor import Extractor, ExtractorFactory
from .processor import Processor, ProcessorFactory
from .writer import Writer, WriterFactory

__all__ = ["Pipeline"]


def _extract_worker(stage, src: IterableQueue[Path], dst: IterableQueue[Envelope], scratch: str, index: int, total: int, error, errors) -> None:
    try:
        stage.attach(StageContext(src=src, dst=dst, scratch=Path(scratch), index=index, total=total))
        stage.execute()

    except BaseException as e:
        errors.put(f"{type(e).__name__}: {e}")
        error.set()

    finally:
        try:
            stage.close()
        finally:
            dst.done()


def _process_worker(stages: list[Processor], src: IterableQueue[Envelope], dst: IterableQueue[Envelope], scratch: str, base_index: int, total: int, error, errors) -> None:
    try:
        # internal, non-mp queues chaining the processors
        internal: list[IterableQueue[Envelope]] = [IterableQueue() for _ in range(len(stages) + 1)]

        # first in series consumes from extractor, last feeds writer
        internal[0] = src
        internal[-1] = dst

        # start each stage
        threads: list[Thread] = []
        for j, s in enumerate(stages):
            s.attach(StageContext(
                src=internal[j], dst=internal[j + 1], scratch=Path(scratch), index=base_index + j, total=total
            ))
            threads.append(Thread(target=s.execute, daemon=True))

        for t in threads:
            t.start()

        for t in threads:
            t.join()

    except BaseException as e:
        errors.put(f"{type(e).__name__}: {e}")
        error.set()

    finally:
        try:
            for s in stages:
                s.close()
        finally:
            dst.done()


def _write_worker(stage, src: IterableQueue[Envelope], dst: IterableQueue[Envelope], scratch: str, index: int, total: int, error, errors) -> None:
    try:
        stage.attach(StageContext(src=src, dst=dst, scratch=Path(scratch), index=index, total=total))
        stage.execute()

    except BaseException as e:
        errors.put(f"{type(e).__name__}: {e}")
        error.set()

    finally:
        try:
            stage.close()
        finally:
            dst.done()


class Pipeline:
    """
    Concurrent, process-based data processing pipeline.

    Extractor procs -> [mp queue] -> processor procs (multithreaded) -> [mp queue] -> writer procs -> [mp queue] -> tracker.
    Use as a context manager to guarantee finalization.
    """

    def __init__(
            self,
            source: Source | SourceType,
            extractor: Extractor[Any],
            processors: list[Processor[Any]],
            writer: Writer[Any]
    ) -> None:
        source = Source(source)

        self._extractor = extractor
        self._processors = processors
        self._writer = writer

        # resolve files eagerly; extractors consume them from an mp queue
        self._files: list[Path] = list(source.resolve(getattr(extractor, "file_ext")))
        self._file_count = len(self._files)

        self._scratch = tempfile.TemporaryDirectory(prefix="icegraph_")
        self._procs: list[mp.Process] = []

    @classmethod
    def from_yaml(cls, source: Source | SourceType, config_path: str | Path) -> Self:
        with Path(config_path).open("r") as f:
            config = Config(**yaml.safe_load(f))

        stage_config = config.extractor
        extractor = ExtractorFactory.create(stage_config.name, **stage_config.kwargs)

        processors = []
        for stage_config in config.processors:
            processors.append(ProcessorFactory.create(stage_config.name, **stage_config.kwargs))

        stage_config = config.writer
        writer = WriterFactory.create(stage_config.name, **stage_config.kwargs)

        return cls(source, extractor, processors, writer)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    ### EXECUTOR

    def _split_procs(self, nproc: int, ratio: tuple[int, int, int]) -> tuple[int, int, int]:
        if nproc < 3:
            raise ValueError("nproc must be >= 3.")

        total = sum(ratio)
        counts = [max(1, (nproc * r) // total) for r in ratio]

        # any extra procs not allocated above get distributed among lowest-proc-count phases first
        # for ties, fills left to right round-robin
        while sum(counts) < nproc:
            counts[counts.index(min(counts))] += 1

        # while the floor div cant actually make sum(counts) > nproc, the max(1, ...) can
        # for example with nproc = 3 and ratio = (100, 1, 1), we get counts = (2, 1, 1)
        # in this case and in similar cases sum(counts) > nproc and must be trimmed
        while sum(counts) > nproc:
            counts[counts.index(max(counts))] -= 1

        return cast(tuple[int, int, int], tuple(counts))

    def execute(
            self, *,
            nproc: int = 3,
            epw_ratio: tuple[int, int, int] = (3, 1, 3)
    ) -> dict[str, float]:
        """
        Launch all stage processes. Blocks until completion or failure.
        Returns averaged metrics recorded over the run.
        """
        error = mp.Event()
        errors: mp.Queue = mp.Queue()

        total = 1 + len(self._processors) + 1
        scratch = self._scratch.name

        procs = self._split_procs(nproc, epw_ratio)

        # channels between process groups (parent is sole producer of files)
        files_ch    = IterableQueue(mp=True, producers=1,           consumers=procs[0], maxsize=0)
        ex_ch       = IterableQueue(mp=True, producers=procs[0],    consumers=procs[1])
        pr_ch       = IterableQueue(mp=True, producers=procs[1],    consumers=procs[2])
        out_ch      = IterableQueue(mp=True, producers=procs[2],    consumers=1)

        self._procs = []
        for n in range(procs[0]):
            self._procs.append(mp.Process(
                target=_extract_worker,
                args=(self._extractor, files_ch, ex_ch, scratch, 0, total, error, errors),
                name=f"pipeline-extract-{n}", daemon=True
            ))
        for n in range(procs[1]):
            self._procs.append(mp.Process(
                target=_process_worker,
                args=(self._processors, ex_ch, pr_ch, scratch, 1, total, error, errors),
                name=f"pipeline-process-{n}", daemon=True
            ))
        for n in range(procs[2]):
            self._procs.append(mp.Process(
                target=_write_worker,
                args=(self._writer, pr_ch, out_ch, scratch, total - 1, total, error, errors),
                name=f"pipeline-write-{n}", daemon=True
            ))

        for p in self._procs:
            p.start()

        # feed source files, then signal end of input
        for path in self._files:
            files_ch.put(path)
        files_ch.done()

        # track writer output for progress and metrics
        metrics = self.track(out_ch)

        for p in self._procs:
            p.join()

        if error.is_set():
            msg = errors.get() if not errors.empty() else "unknown stage failure"
            raise RuntimeError(f"Pipeline stage crashed: {msg}")

        return metrics

    def track(self, out_ch: IterableQueue[Envelope]) -> dict[str, float]:
        progress = Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            TextColumn("| ETA:"),
            TimeRemainingColumn(),
            transient=False,
            expand=True,
            console=console,
            refresh_per_second=10,
            speed_estimate_period=300.0
        )

        with progress:
            task = progress.add_task("Processing", total=self._file_count)

            metrics: dict[str, float] = {}
            count = 0
            for item in out_ch:
                count += 1
                for key, value in item.metrics.items():
                    metrics[key] = metrics.get(key, 0.0) + (value - metrics.get(key, 0.0)) / count
                progress.advance(task)

        return metrics

    def close(self) -> None:
        """Terminate live workers and clean temporary resources."""
        # kill all children
        for p in self._procs:
            if p.is_alive():
                p.terminate()

        for p in self._procs:
            p.join(timeout=5)

        self._scratch.cleanup()
