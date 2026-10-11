# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from threading import Thread
from typing import Self, Any, cast
from pathlib import Path
import tempfile
import multiprocessing as mp
from multiprocessing.process import BaseProcess
from multiprocessing.synchronize import Event

import yaml

from icegraph.common.files import SourceType, Source
from icegraph.utils import set_proctitle

from .shared.queue import IterableQueue
from .config import Config
from .types import StageContext, Envelope
from .console import Tally, Step, PipelineConsole

from .extractor import Extractor, ExtractorFactory
from .collector import Collector, CollectorConfig
from .processor import Processor, ProcessorFactory
from .writer import Writer, WriterFactory

import logging
logger = logging.getLogger(__name__)

__all__ = ["Pipeline"]

# polars thread pool is not fork-safe, forked children can deadlock on frame ops
_MP_CTX = mp.get_context("spawn")


def _extract_worker(
        stage: Extractor,
        src: IterableQueue[Path],
        dst: IterableQueue[Envelope],
        tally: Tally,
        scratch: str,
        stage_count: int,
        error: Event,
        errors: mp.Queue,
        worker_index: int
) -> None:
    set_proctitle(f"icegraph-extractor-{worker_index}")

    # allow main process to orchestrate shutdown on interrupt
    import signal, sys
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    try:
        stage.attach(StageContext(src=tally.count(src, 0), dst=dst, scratch=Path(scratch), index=0, total=stage_count))
        stage.execute()

    except BaseException as e:
        errors.put(f"{type(e).__name__}: {e}")
        error.set()

    finally:
        try:
            stage.close()
        finally:
            dst.done()


def _collect_worker(
        stage: Collector,
        src: IterableQueue[Envelope],
        dst: IterableQueue[Envelope],
        tally: Tally,
        scratch: str,
        stage_count: int,
        error: Event,
        errors: mp.Queue,
        worker_index: int
) -> None:
    set_proctitle(f"icegraph-collector-{worker_index}")

    # allow main process to orchestrate shutdown on interrupt
    import signal, sys
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    try:
        stage.attach(StageContext(src=tally.count(src, 1), dst=dst, scratch=Path(scratch), index=1, total=stage_count))
        stage.execute()

    except BaseException as e:
        errors.put(f"{type(e).__name__}: {e}")
        error.set()

    finally:
        try:
            stage.close()
        finally:
            dst.done()


def _process_worker(
        stages: list[Processor],
        src: IterableQueue[Envelope],
        dst: IterableQueue[Envelope],
        tally: Tally,
        scratch: str,
        stage_count: int,
        error: Event,
        errors: mp.Queue,
        worker_index: int
) -> None:
    set_proctitle(f"icegraph-processor-{worker_index}")

    # allow main process to orchestrate shutdown on interrupt
    import signal, sys
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    def _run(stage: Processor, out_q: IterableQueue[Envelope]) -> None:
        # report stage-thread failures to the parent instead of dying silently
        try:
            stage.execute()

        except SystemExit:
            raise

        except BaseException as e:
            errors.put(f"{type(e).__name__}: {e}")
            error.set()

        finally:
            # close this stage's internal outbound queue so the next stage in the
            # chain sees end-of-input and exits, cascading shutdown to the tail
            # the shared `dst` is closed once per process, in the finally below
            if out_q is not dst:
                out_q.done()

    internal: list[IterableQueue[Envelope]] = []
    try:
        # internal, non-mp queues chaining the processors
        internal = [IterableQueue(maxsize=1) for _ in range(len(stages) + 1)]

        # first in series consumes from extractor, last feeds writer
        internal[0] = src
        internal[-1] = dst

        # start each stage
        threads: list[Thread] = []
        for j, s in enumerate(stages):
            s.attach(StageContext(
                src=tally.count(internal[j], 2 + j), dst=internal[j + 1], scratch=Path(scratch), index=2 + j, total=stage_count
            ))
            threads.append(Thread(target=_run, args=(s, internal[j + 1]), daemon=True))

        for t in threads:
            t.start()

        for t in threads:
            t.join()

    except SystemExit:
        raise

    except BaseException as e:
        errors.put(f"{type(e).__name__}: {e}")
        error.set()

    finally:
        try:
            for s in stages:
                s.close()
        finally:
            dst.done()


def _write_worker(
        stage: Writer,
        src: IterableQueue[Envelope],
        dst: IterableQueue[Envelope],
        tally: Tally,
        scratch: str,
        stage_count: int,
        error: Event,
        errors: mp.Queue,
        worker_index: int,
        outdir: Path
) -> None:
    set_proctitle(f"icegraph-writer-{worker_index}")

    # allow main process to orchestrate shutdown on interrupt
    import signal, sys
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    try:
        stage.attach(
            StageContext(
                src=tally.count(src, stage_count - 1),
                dst=dst,
                scratch=Path(scratch),
                index=stage_count - 1,
                total=stage_count,
                outdir=outdir
            )
        )
        stage.execute()

    except SystemExit:
        raise

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

    Extractor procs -> [mp queue] -> collector proc -> [mp queue] -> processor procs (multithreaded) -> [mp queue]
    -> writer procs -> [mp queue] -> tracker. The collector runs in one process on top of ``nproc``.
    Use as a context manager to guarantee finalization.
    """

    def __init__(
            self,
            source: Source | SourceType,
            outdir: str | Path,
            extractor: Extractor[Any],
            processors: list[Processor[Any]],
            writer: Writer[Any],
            collector: Collector
    ) -> None:
        source = Source(source)

        # cache outdir to pass to writers
        self._outdir = Path(outdir)

        self._extractor = extractor
        self._collector = collector
        self._processors = processors
        self._writer = writer

        # resolve files eagerly; extractors consume them from an mp queue
        self._files: list[Path] = list(source.resolve(getattr(extractor, "file_ext")))
        self._file_count = len(self._files)

        # a file read twice would count twice toward its set
        if len({path.resolve() for path in self._files}) != self._file_count:
            raise ValueError("The source names at least one file more than once.")

        self._scratch:  tempfile.TemporaryDirectory = tempfile.TemporaryDirectory(prefix="icegraph_")
        self._procs:    list[BaseProcess]           = []
        self._channels: list[IterableQueue[Any]]    = []
        self._error:    Event | None                = None
        self._errors:   mp.Queue[Any] | None        = None

    @classmethod
    def from_yaml(cls, source: Source | SourceType, outdir: str | Path, config_path: str | Path) -> Self:
        with Path(config_path).open("r") as f:
            config = Config(**yaml.safe_load(f))

        stage_config = config.extractor
        extractor = ExtractorFactory.create(stage_config.name, **stage_config.kwargs)

        collector = Collector(config.collector)

        processors = []
        for stage_config in config.processors:
            processors.append(ProcessorFactory.create(stage_config.name, **stage_config.kwargs))

        stage_config = config.writer
        writer = WriterFactory.create(stage_config.name, **stage_config.kwargs)

        return cls(source, outdir, extractor, processors, writer, collector)

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
        self._error = _MP_CTX.Event()
        self._errors = _MP_CTX.Queue()

        stage_count = 1 + 1 + len(self._processors) + 1
        scratch = self._scratch.name

        # every stage counts the items passing through it, for the console
        tally = Tally(stage_count, _MP_CTX)

        procs = self._split_procs(nproc, epw_ratio)

        # shards are numbered from 0 every run, so ones already there may be overwritten or mixed in
        if self._outdir.is_dir() and any(self._outdir.iterdir()):
            logger.warning(f"output directory {self._outdir} is not empty, files there may be overwritten")

        # channels between process groups (parent is sole producer of files)
        self._channels = [
            IterableQueue(mp=True, ctx=_MP_CTX, producers=1,        consumers=procs[0], maxsize=0       ),
            IterableQueue(mp=True, ctx=_MP_CTX, producers=procs[0], consumers=1,        maxsize=procs[0]),
            IterableQueue(mp=True, ctx=_MP_CTX, producers=1,        consumers=procs[1], maxsize=procs[1]),
            IterableQueue(mp=True, ctx=_MP_CTX, producers=procs[1], consumers=procs[2], maxsize=procs[2]),
            IterableQueue(mp=True, ctx=_MP_CTX, producers=procs[2], consumers=1,        maxsize=1       )
        ]

        # the console draws each stage with the queue feeding it, in stage order
        # processors past the first are fed by queues internal to each processor process
        steps = [
            Step("extract", self._extractor.name, procs[0], self._channels[0]),
            Step("collect", self._collector.name, 1, self._channels[1]),
            *(Step("process", p.name, procs[1], None if j else self._channels[2]) for j, p in enumerate(self._processors)),
            Step("write", self._writer.name, procs[2], self._channels[3])
        ]

        self._procs = []
        for n in range(procs[0]):
            self._procs.append(_MP_CTX.Process(
                target=_extract_worker,
                args=(self._extractor, self._channels[0], self._channels[1], tally, scratch, stage_count, self._error, self._errors, n),
                name=f"icegraph-extractor-{n}", daemon=True
            ))

        # only ever need one collector, it only does a mapping no actual work
        self._procs.append(_MP_CTX.Process(
            target=_collect_worker,
            args=(self._collector, self._channels[1], self._channels[2], tally, scratch, stage_count, self._error, self._errors, 0),
            name="icegraph-collector", daemon=True
        ))

        for n in range(procs[1]):
            self._procs.append(_MP_CTX.Process(
                target=_process_worker,
                args=(self._processors, self._channels[2], self._channels[3], tally, scratch, stage_count, self._error, self._errors, n),
                name=f"icegraph-processor-{n}", daemon=True
            ))

        for n in range(procs[2]):
            self._procs.append(_MP_CTX.Process(
                target=_write_worker,
                args=(self._writer, self._channels[3], self._channels[4], tally, scratch, stage_count, self._error, self._errors, n, self._outdir),
                name=f"icegraph-writer-{n}", daemon=True
            ))

        for p in self._procs:
            p.start()

        # feed source files, then signal end of input
        for path in self._files:
            self._channels[0].put(path)
        self._channels[0].done()

        # track writer output for progress and metrics
        view = PipelineConsole(self._outdir, self._file_count, steps, tally)
        metrics = self.track(self._channels[4], view, self._error)

        if self._error.is_set():
            # drain the error channel before close() tears it down
            message = self._first_error(self._errors)

            # a stage died, surviving workers are blocked on queues that will
            # never be fed again, so stop them rather than join forever
            self.close()

            raise RuntimeError(f"Pipeline stage crashed: {message}")

        for p in self._procs:
            p.join(timeout=30)

        if any(p.is_alive() for p in self._procs):
            logger.warning("workers did not exit cleanly, terminating")

        self.close()
        return metrics

    @staticmethod
    def _first_error(errors: mp.Queue) -> str:
        """Drain the error channel, returning the first reported failure."""
        messages: list[str] = []
        while True:
            try:
                # brief timeout, a crashing child may still be flushing its message
                messages.append(errors.get(timeout=0.5))
            except Exception:
                break

        if not messages:
            return "unknown stage failure"

        # keep the rest for debugging, the first is the root cause
        for msg in messages[1:]:
            logger.debug("additional stage failure: %s", msg)

        return messages[0]

    def track(
            self,
            out_ch: IterableQueue[Envelope],
            view: PipelineConsole,
            error: Event | None = None
    ) -> dict[str, float]:
        # the view reads the workers' counts as it refreshes, so only written shards need passing on
        with view:
            timeout = 0.5
            while True:
                if error is not None and error.is_set():
                    # on error, drain quickly
                    timeout = 0.0

                # a crashed stage never emits its sentinel, so poll instead of
                # blocking forever waiting on output that will never arrive
                try:
                    item = out_ch.poll(timeout)
                except StopIteration:
                    break

                # nothing ready yet, re-check for failures
                if item is None:
                    if error is not None and error.is_set():
                        break
                    if view.files >= self._file_count:
                        logger.warning("all %d files received but no sentinel arrived", view.files)
                        break
                    continue

                # data is persisted, drop the scratch arrow files so scratch space stays bounded
                item.quiver.close()
                view.add(item)

        return view.metrics

    def close(self) -> None:
        """Terminate live workers and clean temporary resources."""
        for p in self._procs:
            if p.is_alive():
                p.terminate()

        for p in self._procs:
            p.join(timeout=5)

            if p.is_alive():
                logger.warning("worker %s ignored terminate, killing", p.name)
                p.kill()
                p.join(timeout=2)

        for channel in self._channels:
            channel.terminate()
        self._channels = []

        if self._errors is not None:
            self._errors.close()
            self._errors.cancel_join_thread()
            self._errors = None

        # remove ref to error event
        self._error = None

        self._scratch.cleanup()
