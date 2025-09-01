# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

# first export os var to ensure hdf5 file locking
import os
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "TRUE")

from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, ClassVar, Optional, Iterator, Sequence, Union, TypeVar, Tuple, \
    TypeAlias, Self, TYPE_CHECKING, Dict, Any, Type
from queue import Queue, Empty, Full
from pathlib import Path
import weakref
import shutil
import tempfile
import threading
import os
from threading import Thread, current_thread, Event
import logging

import pandas as pd
import portalocker

from .extractor.base import Extractor
from .processor.base import Processor
from .writers.base import Writer
from icegraph.utils.pathutils import PathResolver
from icegraph.console import Console
from .base.exceptions import PipelineBuildError
from icegraph.config import IGConfig

import faulthandler, signal

# Enable built-in fatal-signal handling (covers SIGSEGV etc.)
faulthandler.enable(all_threads=True)

# Optional: also dump stacks on SIGUSR2 / SIGTERM (do NOT try SIGSEGV here)
for sig_name in ("SIGUSR2", "SIGTERM"):
    sig = getattr(signal, sig_name, None)
    if sig is None:
        continue
    try:
        faulthandler.register(sig, all_threads=True, chain=True)
    except (RuntimeError, ValueError):
        # Ignore if not supported on this platform or already handled
        pass

if TYPE_CHECKING:
    from .base.operator import Operator
else:
    Operator = None


T = TypeVar("T", bound="Operator")

SentinelType: TypeAlias = Tuple[str, str]
EnvelopeOrSentinel: TypeAlias = Union["Pipeline.Envelope", SentinelType]

def nested_dict():
    return defaultdict(nested_dict)

class Pipeline:
    """Initialize the data processing pipeline. MUST BE USED AS A CONTEXT MANAGER TO ENSURE PROPER CLEANUP."""

    SENTINEL: ClassVar[SentinelType] = ("__END__", "Pipeline")

    # constants
    MAX_QUEUE_SIZE: ClassVar[int]   = 5
    TIMEOUT:        ClassVar[float] = 1.0

    # full hdf5 thread lock
    HDF5_LOCK = threading.Lock()

    @dataclass
    class FileHandle:
        """
        A closeable handle to a file. On `close()` (or context exit), the file is deleted.
        Also tries to remove the parent directory if it becomes empty.

        Works within a single owner/stage.
        """
        src: Path
        _finalizer: weakref.finalize = field(init=False, repr=False)

        FILELOCK_SH_TIMEOUT: ClassVar[int] = 30
        FILELOCK_EX_TIMEOUT: ClassVar[int] = 60

        def __post_init__(self) -> None:
            self._finalizer = weakref.finalize(self, Pipeline.FileHandle._cleanup, self.src)

        # ---- Locks (sidecar .lock file) ----
        @property
        def _lock_path(self) -> Path:
            return self.src.with_suffix(self.src.suffix + ".lock")

        def lock_shared(self, timeout: Union[int, float] = FILELOCK_SH_TIMEOUT):
            """Readers: allow many; blocks if a writer holds the lock."""
            return portalocker.Lock(
                self._lock_path, timeout=timeout, flags=portalocker.LockFlags.SHARED | portalocker.LockFlags.NON_BLOCKING
            )

        def lock_exclusive(self, timeout: Union[int, float] = FILELOCK_EX_TIMEOUT):
            """Writers: exclusive; blocks until all readers/writers are done."""
            return portalocker.Lock(
                self._lock_path, timeout=timeout, flags=portalocker.LockFlags.EXCLUSIVE | portalocker.LockFlags.NON_BLOCKING
            )

        # ---- Cleanup ----
        def close(self) -> None:
            if self._finalizer.alive:
                self._finalizer()

        @staticmethod
        def _cleanup(path: Path) -> None:
            try:
                path.unlink(missing_ok=True)
            except IsADirectoryError:
                shutil.rmtree(path, ignore_errors=True)
            # remove sidecar lock file too
            try:
                path.with_suffix(path.suffix + ".lock").unlink(missing_ok=True)
            except Exception:
                pass

    @dataclass
    class Envelope:
        """Envelope containing data passed through the pipeline."""
        df: pd.DataFrame
        fh: Pipeline.FileHandle
        attrs: Dict[str, Dict[str, Any]] = field(default_factory=nested_dict)

    def __init__(self):
        # warm operators
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

        # set up working directory
        tmp = tempfile.TemporaryDirectory(
            prefix="icegraph_", dir="/dev/shm" if os.path.isdir("/dev/shm") else None
        )

        self._local_working_dir = tmp
        self.local_working_dir_path = Path(tmp.name)

        # set the multiprocess-safe working directory
        home = Path(os.path.expanduser("~"))
        self._global_working_dir = home / ".cache"

        # grab global config
        self._config: IGConfig = IGConfig.get()

    def __enter__(self) -> Self:
        Console.out("Initializing IceGraph data processing pipeline...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.terminate()

    def terminate(self) -> None:
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

        # clean the working directory
        try:
            self._local_working_dir.cleanup()
        except Exception:
            shutil.rmtree(self.local_working_dir_path, ignore_errors=True)

    ### PORTS

    def iter_output(self) -> Iterator[pd.DataFrame]:
        """
        Consume the pipeline's output queue and yield DataFrames as they arrive.
        Stops on sentinel or when self.stop is set.
        """
        if not self.queues:
            raise RuntimeError("Pipeline not configured.")

        iter_queue = self._iter_from_queue(self.queues[-1], self.stop)

        for item in iter_queue:
            try:
                # Close the temp file under locks, then yield the DataFrame
                with item.fh.lock_exclusive(), self.HDF5_LOCK:
                    item.fh.close()
                yield item

            finally:
                # Mark the exact queue item as done
                self.queues[-1].task_done()

    def start_output_printer(self, *, name="output-printer") -> Thread:
        """Spawn a daemon thread that prints items for debugging."""

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

    def configure(self, source: Union[str, Path, Sequence[Union[str, Path]]], outdir: Optional[Union[str, Path]] = None) -> None:
        if not self._build_called:
            raise RuntimeError("Must call build() before configure().")

        self.source = source
        self.file_list = tuple(PathResolver.normalize_sources(
            source, ".hdf5" if self._extractor is None else ".i3.zst"
        ))

        if outdir is not None:
            self.outdir = Path(outdir)
            self.outdir.mkdir(exist_ok=True, parents=True)

        # wire stages between each other
        self._wire_stages()

        # flip flag
        self._configure_called = True

    def _check_pre_reqs(self) -> None:
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
                f"has prerequisites {req_repr}"
            )
            for required in pre_reqs:
                if isinstance(required, tuple):
                    if not any(r in seen_specs for r in required):
                        raise _BuildError
                elif required not in seen_specs:
                    raise _BuildError

    def _wire_stages(self) -> None:
        if self.file_list is None:
            raise RuntimeError("file_list not set; call configure().")

        stage_count = len(self._stage_specs)
        if stage_count == 0:
            raise RuntimeError("No stages to wire.")

        # create one outbound queue per stage
        self.queues = [Queue(maxsize=type(self).MAX_QUEUE_SIZE) for _ in range(stage_count)]

        # instantiate stages with appropriate input iterators
        for i, spec in enumerate(self._stage_specs):
            in_iter: Optional[Union[Iterator[Pipeline.Envelope], Iterator[Path]]]
            stage = spec()

            if i == 0:
                # Seed first stage by calling its bootstrap on each file AFTER instantiation.
                # create a generator bound to this stage instance
                in_iter = self.seed_iter(stage)
            else:
                # downstream stages consume from previous queue
                in_iter = self._iter_from_queue(self.queues[i - 1], stop=self.stop)

            stage.set_in_iter(in_iter)
            stage.assign_queue(i)
            stage.set_parent(self)

            self._stages.append(stage)

    ### EXECUTOR

    def execute(self, *, debug: bool = False) -> None:
        if not self._configure_called:
            raise RuntimeError("Call configure() before execute().")
        if self._threads:
            raise RuntimeError("execute() already called")

        def _runner(stage):
            try:
                stage.execute()
            except BaseException as e:
                logging.error(
                    "Stage crashed: %s", current_thread().name, exc_info=(type(e), e, e.__traceback__)
                )
                # signal everyone to stop
                self.terminate()
                raise

        # start
        for i, s in enumerate(self._stages):
            t = Thread(
                target=_runner, args=(s,), name=f"pipeline-stage-{i}-{type(s).__name__}", daemon=True
            )
            t.start()
            self._threads.append(t)

        # start test printer
        if debug is True:
            printer_thread = self.start_output_printer()
            self._threads.append(printer_thread)

        # use writer if debug is false and there is a writer
        if self._writer is not None and debug is False:
            writer_thread = self._start_writer()
            self._threads.append(writer_thread)

        for thread in self._threads:
            thread.join()

    ### WRITER

    def _start_writer(self) -> Thread:

        def _runner(outdir: Optional[Path]) -> None:
            # grab outdir location
            outdir = Path(outdir or self._config.user_config.io.default_dir)
            try:
                for item in Console.progress_bar(self.iter_output(), total=len(self.file_list)):
                    # dynamically generate output file path
                    outfile = outdir / item.fh.src.with_suffix(self._writer.suffix).name
                    # write to the file
                    with self._writer(outfile) as writer:
                        writer.write_attrs(item.attrs)
                        writer.write(item.df)

            except BaseException as e:
                logging.error(
                    "Stage crashed: %s", current_thread().name, exc_info=(type(e), e, e.__traceback__)
                )
                # signal everyone to stop
                self.terminate()
                raise

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
        assert self.file_list is not None
        for path in self.file_list:
            if self.stop.is_set():
                break
            env = s.bootstrap(path)
            if env is not None:
                yield env

    @classmethod
    def _iter_from_queue(cls, in_queue: Queue[EnvelopeOrSentinel], stop: Optional[Event] = None) -> Iterator[Envelope]:
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