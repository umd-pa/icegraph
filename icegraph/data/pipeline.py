# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

# first export os var to ensure hdf5 file locking
import os
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "TRUE")

from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, ClassVar, Optional, Iterator, Sequence, Union, Tuple, \
    TypeAlias, Self, Dict, Any, Type
from queue import Queue, Empty, Full
from pathlib import Path
import weakref
import shutil
import tempfile
import time
import threading
from threading import Thread, current_thread, Event
import logging
import functools

import pandas as pd
import portalocker

from .extractor.base import Extractor
from .processor.base import Processor
from .writers.base import Writer
from icegraph.utils.pathutils import PathResolver
from icegraph.console import Console
from .base.exceptions import PipelineBuildError
from icegraph.config import IGConfig
from icegraph.utils.mputils import MPTempDir

import faulthandler, signal

# Enable built-in fatal-signal handling (covers SIGSEGV etc.)
faulthandler.enable(all_threads=True)

# also dump stacks on SIGUSR2 / SIGTERM (do NOT try SIGSEGV here)
for sig_name in ("SIGUSR2", "SIGTERM"):
    sig = getattr(signal, sig_name, None)
    if sig is None:
        continue
    try:
        faulthandler.register(sig, all_threads=True, chain=True)
    except (RuntimeError, ValueError):
        # Ignore if not supported on this platform or already handled
        pass

SentinelType: TypeAlias = Tuple[str, str]
EnvelopeOrSentinel: TypeAlias = Union["Pipeline.Envelope", SentinelType]
StageSpec: TypeAlias = Union[Extractor, Processor, Writer]

def nested_dict():
    return defaultdict(nested_dict)


class Pipeline:
    """
    Concurrent, stage-based data processing pipeline.

    This pipeline wires an optional Extractor, a sequence of Processor stages,
    and an optional Writer. Stages communicate via bounded queues, and the
    pipeline coordinates startup, teardown, and resource cleanup.
    Use as a context manager to guarantee finalization.
    """

    SENTINEL: ClassVar[SentinelType] = ("__END__", "Pipeline")

    # constants
    MAX_QUEUE_SIZE: ClassVar[int]   = 5
    TIMEOUT:        ClassVar[float] = 1.0

    # full hdf5 thread lock
    HDF5_LOCK = threading.Lock()

    def __init__(self):
        """Construct an empty pipeline with default queues and temp dirs."""
        # warm stages
        self._extractor:    Optional[Type[Extractor]]         = None
        self._processors:   Optional[List[Type[Processor]]]   = None
        self._writer:       Optional[Type[Writer]]            = None

        # warm file list
        self.file_list: Optional[Tuple[Path, ...]]                              = None
        self.source:    Optional[Union[str, Path, Sequence[Union[str, Path]]]]  = None
        self.outdir:    Optional[Path]                                          = None

        # setup queues
        self.queues: List[Queue[EnvelopeOrSentinel]] = []

        # warm stage specs and stages
        self._stage_specs:  List[Union[Type[Extractor], Type[Processor]]]   = []
        self._stages:       List[Union[Extractor, Processor]]               = []

        # thread pool
        self._threads:  List[Thread]    = []
        self.stop:      Event           = Event()

        # flags
        self._build_called:     bool = False
        self._configure_called: bool = False

        # set up local and global working directories
        self.local_tmpdir_obj = tempfile.TemporaryDirectory(
            prefix="icegraph_", dir="/dev/shm" if os.path.isdir("/dev/shm") else None
        )
        self.global_tmpdir_obj = MPTempDir()

        # normalize to paths
        self.local_working_dir = Path(self.local_tmpdir_obj.name)
        self.global_working_dir = Path(self.global_tmpdir_obj.tempdir)

        # grab global config
        self._config: IGConfig = IGConfig.get()

    def __enter__(self) -> Self:
        Console.out("Initializing IceGraph data processing pipeline...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.terminate()

    def terminate(self) -> None:
        """Signal stages to stop, flush queues, and clean temporary resources."""
        Console.out("Terminating pipeline and running cleanup, this may take a few seconds...", severity=2)
        self.stop.set()
        for q in self.queues:
            try:
                q.put_nowait(self.SENTINEL)
            except Full:
                # if full, rely on stage finally blocks to propagate
                pass

        # execute stop process on children if implemented
        if self._stages:
            for stage in self._stages:
                if hasattr(stage, "close"):
                    stage.close()

        # clean the working directories
        try:
            self.local_tmpdir_obj.cleanup()
            self.global_tmpdir_obj.terminate_instance()
        except Exception:
            shutil.rmtree(str(self.local_working_dir), ignore_errors=True)
            shutil.rmtree(str(self.global_working_dir), ignore_errors=True)

    ### PORTS

    def iter_output(self) -> Iterator[Envelope]:
        """
        Yield envelopes from the final stage, performing safe cleanup. Yields until a sentinel
        item arrives or the stop event is set.
        """
        if not self.queues:
            raise RuntimeError("Pipeline not configured.")

        iter_queue = self._iter_from_queue(self.queues[-1], self.stop)

        # item will always be an envelope for final queue
        item: Pipeline.Envelope
        for item in iter_queue:
            try:
                with self.HDF5_LOCK:
                    yield item.finalize()
            finally:
                self.queues[-1].task_done()


    def start_output_printer(self, *, name="output-printer") -> Thread:
        """
        Spawn a daemon thread that prints envelope dataframes for debugging.

        Args:
            name: Thread name.
        Returns:
            The started Thread object.
        """

        def _printer():
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            for item in self.iter_output():
                print(item.df)

        t = Thread(target=_printer, name=name, daemon=True)
        t.start()
        return t

    ### BUILD/CONFIGURE

    def build(
        self, *,
        extractor: Optional[Type[Extractor]] = None,
        processors: List[Type[Processor]],
        writer: Optional[Type[Writer]] = None
    ) -> None:
        """
        Define the stage types (Extractor/Processors/Writer) and validate order.

        Must be called before :meth:`configure`.
        """
        self._extractor = extractor
        self._processors = processors
        self._writer = writer

        # build stages
        self._stage_specs = processors if extractor is None else [extractor, *processors]

        # check for compatibility and ordering
        self._check_pre_reqs()

        # flip flag
        self._build_called = True
        # if build is called more than once, ensure configure is called again before execution
        self._configure_called = False

    def configure(
            self,
            source: Union[str, Path, Sequence[Union[str, Path]]],
            outdir: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Bind sources, normalize file list, create output directory, and wire stages.

        Args:
            source: Single path, glob, or sequence of paths.
            outdir: Optional output directory for the Writer.
        Raises:
            RuntimeError: If :meth:`build` has not been called.
        """
        if not self._build_called:
            raise RuntimeError("Must call build() before configure().")

        self.source = source
        self.file_list = tuple(PathResolver.normalize_sources(
            source, ".hdf5" if self._extractor is None else ".i3.zst"
        ))

        # console output
        Console.out(f"Loading from source: {Console.source_repr(self.source)}")

        if outdir is not None:
            self.outdir = Path(outdir)
            self.outdir.mkdir(exist_ok=True, parents=True)

        # wire stages between each other
        self._wire_stages()

        # flip flag
        self._configure_called = True

    def _check_pre_reqs(self) -> None:
        """
        Validate processor prerequisite ordering.

        Raises:
            PipelineBuildError: If a stage's prerequisites are not satisfied.
        """
        seen_specs = []
        for spec in self._stage_specs:
            # append to seen list, we can do this before checks without problems
            seen_specs.append(spec)

            # only processors have prereqs
            if not issubclass(spec, Processor):
                continue
            pre_reqs = spec.PRE_REQS

            # no prereqs, continue
            if pre_reqs is None:
                continue

            # build a string repr for printout
            _req_repr_list: List[str] = []
            for required in pre_reqs:
                if isinstance(required, tuple):
                    _req_repr_list.append(f"({' or '.join([req.__name__ for req in required])})")
                else:
                    _req_repr_list.append(required.__name__)
            req_repr = f"[{', '.join(_req_repr_list)}]"

            # check if the required prereqs are present, if not raise
            _BuildError = PipelineBuildError(
                f"Error building pipeline, processor module {spec.__name__} "
                f"has unsatisfied prerequisites {req_repr}"
            )
            for required in pre_reqs:
                if isinstance(required, tuple):
                    if not any(r in seen_specs for r in required):
                        raise _BuildError
                elif required not in seen_specs:
                    raise _BuildError

    def _wire_stages(self) -> None:
        """Instantiate stages, create bounded queues, and connect iterators."""
        if self.file_list is None:
            raise RuntimeError("file_list not set; call configure().")

        stage_count = len(self._stage_specs)
        if stage_count == 0:
            raise RuntimeError("No stages to wire.")

        # One inbound queue per stage + final outbound (stage_count + 1)
        self.queues = [Queue(maxsize=type(self).MAX_QUEUE_SIZE) for _ in range(stage_count + 1)]
        self._stages = []

        for i, spec in enumerate(self._stage_specs):
            # create an instance
            stage = spec()

            # assign the output queue (next from input queue)
            stage.assign_queue(i + 1)

            # tell the stage whos daddy
            stage.set_parent(self)

            # each worker gets its own iterator object that
            # pulls from the *shared* inbound queue self.queues[i]
            if i == 0:
                # first stage consumes Paths and builds envs via its own bootstrap
                in_iter = self._iter_from_bootstrap(self.queues[0], stage, stop=self.stop)
            else:
                # Downstream stages consume envelopes
                in_iter = self._iter_from_queue(self.queues[i], stop=self.stop)

            stage.set_in_iter(in_iter)
            self._stages.append(stage)

    ### EXECUTOR

    def execute(self, *, debug: bool = False) -> None:
        """
        Launch all stage threads and (optionally) the writer/console printer.

        Blocks until all threads finish or an unrecoverable exception occurs.

        Args:
            debug: If True, start the printer instead of the writer.
        Raises:
            RuntimeError: If not configured or already executing.
        """
        if not self._configure_called:
            raise RuntimeError("Call configure() before execute().")
        if self._threads:
            raise RuntimeError("execute() already called")

        @self.runner
        def _runner(stage):
            stage.execute()

        # start
        for i, s in enumerate(self._stages):
            t = Thread(
                target=_runner, args=(s,), name=f"pipeline-stage-{i}-{type(s).__name__}", daemon=True
            )
            t.start()
            self._threads.append(t)

        # start test printer
        if debug:
            printer_thread = self.start_output_printer()
            self._threads.append(printer_thread)

        # use writer if debug is false and there is a writer
        if self._writer is not None and not debug:
            writer_thread = self._start_writer()
            self._threads.append(writer_thread)

        self._threads.append(self._start_feeder(stop=self.stop))

        for thread in self._threads:
            thread.join()

    ### FEEDER

    def _start_feeder(self, *, stop: Optional[Event] = None) -> Thread:
        """
        Feed queue[0] with Paths and then push one sentinel per extractor worker.
        """
        @self.runner
        def _runner(_stop: Optional[Event]) -> None:
            # feed Paths into inbound queue 0
            assert self.file_list is not None
            for path in self.file_list:
                if _stop and _stop.is_set():
                    break
                while True:
                    try:
                        self.queues[0].put(path, timeout=self.TIMEOUT)
                        break
                    except Full:
                        if _stop and _stop.is_set():
                            break
                        continue

            # send one sentinel
            while True:
                try:
                    self.queues[0].put(self.SENTINEL, timeout=self.TIMEOUT)
                    break
                except Full:
                    if _stop and _stop.is_set():
                        # still try to deliver sentinel
                        continue
                    continue

        t = Thread(target=_runner, args=(stop,), name="pipeline-feeder", daemon=True)
        t.start()
        return t

    ### WRITER

    def _start_writer(self) -> Thread:
        """
        Start the writer thread that drains the output and persists results.

        Returns:
            The started writer thread.
        """
        @self.runner
        def _runner(outdir: Optional[Path]) -> None:
            # grab outdir location
            outdir = Path(outdir or self._config.user_config.io.default_dir)
            for item in Console.progress_bar(self.iter_output(), total=len(self.file_list), speed_estimate_period=150):
                # dynamically generate output file path
                outfile = outdir / item.fh.src.with_suffix(self._writer.suffix).name
                # write to the file
                with self._writer(outfile) as writer:
                    writer.write_attrs(item.attrs)
                    writer.write(item.df)

        t = Thread(
            target=_runner,
            args=(self.outdir,),
            name=f"pipeline-stage-{len(self._stages)}-{self._writer.__name__}",
            daemon=True
        )
        t.start()
        return t

    ### HELPERS

    def seed_iter(self, s: Union[Extractor, Processor]) -> Union[Iterator[Path], Iterator[Envelope]]:
        """
        Generator that seeds the first stage (Extractor or Processor).

        Yields each non-None envelope produced by ``s.bootstrap(path)``.
        """
        assert self.file_list is not None
        for path in self.file_list:
            if self.stop.is_set():
                break
            env = s.bootstrap(path)
            if env is not None:
                yield env

    @classmethod
    def _iter_from_queue(
            cls,
            in_queue: Queue[EnvelopeOrSentinel],
            stop: Optional[Event] = None
    ) -> Union[Iterator[Path], Iterator[Envelope]]:
        """
        Pull items from a queue, respecting stop/sentinel semantics.

        Args:
            in_queue: The inbound queue to consume.
            stop: Optional Event to stop consumption early.
        Yields:
            Envelopes until the sentinel is received or stop is set.
        """
        while True:
            if stop is not None and stop.is_set():
                break
            try:
                item = in_queue.get(timeout=cls.TIMEOUT)
            except Empty:
                continue
            if item == cls.SENTINEL:
                break
            yield item

    def _iter_from_bootstrap(
            self,
            in_queue: Queue[Union[Path, EnvelopeOrSentinel]],
            stage: Extractor,
            stop: Optional[Event] = None,
    ) -> Union[Iterator[Path], Iterator[Envelope]]:
        """
        Per-worker iterator: consume Paths from a shared queue and yield Envelopes
        by calling this worker's own `bootstrap(path)`.
        """
        while True:
            if stop is not None and stop.is_set():
                break
            try:
                item = in_queue.get(timeout=self.TIMEOUT)
            except Empty:
                continue
            if item == self.SENTINEL:
                break
            # item is a Path
            env = stage.bootstrap(item)  # each worker uses its own instance here
            if env is not None:
                yield env

    def runner(self, func):
        """Wrapper for thread runners."""
        @functools.wraps(func)
        def inner(*args, **kwargs) -> None:
            try:
                return func(*args, **kwargs)
            except BaseException as e:
                logging.error(
                    "Stage crashed: %s", current_thread().name, exc_info=(type(e), e, e.__traceback__)
                )
                # signal everyone to stop
                self.terminate()
                raise
        return inner
