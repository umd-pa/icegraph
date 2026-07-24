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
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn, TimeElapsedColumn

from icegraph.common.files import SourceType, Source
from icegraph.utils import set_proctitle
from icegraph.ui import console

from .shared.queue import IterableQueue
from .config import Config
from .types import StageContext, Envelope

from .extractor import Extractor, ExtractorFactory
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
        stage.attach(StageContext(src=src, dst=dst, scratch=Path(scratch), index=0, total=stage_count))
        stage.execute()

    except BaseException as e:
        errors.put(f"{type(e).__name__}: {e}")
        error.set()

    finally:
        stage.close()


def _process_worker(
        stages: list[Processor],
        src: IterableQueue[Envelope],
        dst: IterableQueue[Envelope],
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
                src=internal[j], dst=internal[j + 1], scratch=Path(scratch), index=1 + j, total=stage_count
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
        for s in stages:
            s.close()


def _write_worker(
        stage: Writer,
        src: IterableQueue[Envelope],
        dst: IterableQueue[Envelope],
        scratch: str,
        stage_count: int,
        error: Event,
        errors: mp.Queue,
        worker_index: int
) -> None:
    set_proctitle(f"icegraph-writer-{worker_index}")

    # allow main process to orchestrate shutdown on interrupt
    import signal, sys
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    try:
        stage.attach(StageContext(src=src, dst=dst, scratch=Path(scratch), index=stage_count - 1, total=stage_count))
        stage.execute()

    except SystemExit:
        raise

    except BaseException as e:
        errors.put(f"{type(e).__name__}: {e}")
        error.set()

    finally:
        stage.close()


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

        self._scratch:  tempfile.TemporaryDirectory = tempfile.TemporaryDirectory(prefix="icegraph_")
        self._procs:    list[BaseProcess]           = []
        self._channels: list[IterableQueue[Any]]    = []
        self._error:    Event | None                = None
        self._errors:   mp.Queue[Any] | None        = None

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
        self._error = _MP_CTX.Event()
        self._errors = _MP_CTX.Queue()

        stage_count = 1 + len(self._processors) + 1
        scratch = self._scratch.name

        procs = self._split_procs(nproc, epw_ratio)

        # channels between process groups (parent is sole producer of files)
        self._channels = [
            IterableQueue(mp=True, ctx=_MP_CTX, producers=1,        consumers=procs[0], maxsize=0       ),
            IterableQueue(mp=True, ctx=_MP_CTX, producers=procs[0], consumers=procs[1], maxsize=procs[1]),
            IterableQueue(mp=True, ctx=_MP_CTX, producers=procs[1], consumers=procs[2], maxsize=procs[2]),
            IterableQueue(mp=True, ctx=_MP_CTX, producers=procs[2], consumers=1,        maxsize=1       )
        ]

        self._procs = []
        for n in range(procs[0]):
            self._procs.append(_MP_CTX.Process(
                target=_extract_worker,
                args=(self._extractor, self._channels[0], self._channels[1], scratch, stage_count, self._error, self._errors, n),
                name=f"icegraph-extractor-{n}", daemon=True
            ))
        for n in range(procs[1]):
            self._procs.append(_MP_CTX.Process(
                target=_process_worker,
                args=(self._processors, self._channels[1], self._channels[2], scratch, stage_count, self._error, self._errors, n),
                name=f"icegraph-processor-{n}", daemon=True
            ))
        for n in range(procs[2]):
            self._procs.append(_MP_CTX.Process(
                target=_write_worker,
                args=(self._writer, self._channels[2], self._channels[3], scratch, stage_count, self._error, self._errors, n),
                name=f"icegraph-writer-{n}", daemon=True
            ))

        for p in self._procs:
            p.start()

        # feed source files, then signal end of input
        for path in self._files:
            self._channels[0].put(path)
        self._channels[0].done()

        # track writer output for progress and metrics
        metrics = self.track(self._channels[3], self._error)

        if self._error.is_set():
            # a stage died, surviving workers are blocked on queues that will
            # never be fed again, so stop them rather than join forever
            self.close()
            raise RuntimeError(f"Pipeline stage crashed: {self._first_error(self._errors)}")

        for p in self._procs:
            p.join()

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

    def track(self, out_ch: IterableQueue[Envelope], error: Event | None = None) -> dict[str, float]:
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
                    continue

                # data is persisted, drop the scratch arrow files so scratch space stays bounded
                item.quiver.close()

                count += 1
                for key, value in item.metrics.items():
                    metrics[key] = metrics.get(key, 0.0) + (value - metrics.get(key, 0.0)) / count
                progress.advance(task)

        return metrics

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
