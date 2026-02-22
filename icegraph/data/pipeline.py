# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Self, Any
from pathlib import Path
import tempfile
from threading import Thread, current_thread
import logging
import itertools

import yaml
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn, TimeElapsedColumn

from icegraph.types.files import SourceType, Source
from icegraph.ui import console

from .stage import Stage
from .shared.queue import IterableQueue
from .config import Config
from .types import StageContext

from .processor import Processor, ProcessorFactory
from .extractor import Extractor, ExtractorFactory
from .writer import Writer, WriterFactory

__all__ = ["Pipeline"]


class Pipeline:
    """
    Concurrent, stage-based data processing pipeline.

    This pipeline wires an Extractor, a sequence of Processor stages,
    and a Writer. Stages communicate via bounded queues, and the
    pipeline coordinates startup, teardown, and resource cleanup.
    Use as a context manager to guarantee finalization.
    """

    def __init__(
            self,
            source: Source | SourceType,
            extractor: Extractor[Any],
            processors: list[Processor[Any]],
            writer: Writer[Any]
    ) -> None:
        # normalize source
        source = Source(source)

        # cache for source file count, resolved when constructing stages
        self._file_count: int | None = None

        # pack stages to one list
        self._stages: list[Stage] = [extractor, *processors, writer]

        # setup queues, one per stage
        self._queues = [IterableQueue() for _ in range(len(self._stages))]

        # attach each stage
        self._scratch = tempfile.TemporaryDirectory(prefix="icegraph_")
        stage_count = len(self._stages)
        for i, stage in enumerate(self._stages):
            # build the context
            # this will be an extractor for the first stage, so file_ext is defined
            if i == 0:
                files_1, files_2 = itertools.tee(source.resolve(getattr(stage, "file_ext")), 2)

                # use one generator to count files
                self._file_count = sum(1 for _ in files_1)

                # pass the other to the extractor
                src = files_2

            else:
                src = self._queues[i - 1]

            ctx = StageContext(
                src=src, dst=self._queues[i], scratch=Path(self._scratch.name), index=i, total=stage_count
            )

            # attach and pass context
            stage.attach(ctx)

    @classmethod
    def from_yaml(cls, source: Source | SourceType, config_path: str | Path) -> Self:
        with Path(config_path).open("r") as f:
            config = Config(**yaml.safe_load(f))

        # build the extractor (only one)
        stage_config = config.extractor
        extractor = ExtractorFactory.create(stage_config.name, **stage_config.kwargs)

        # build processors
        processors = []
        for stage_config in config.processors:
            processors.append(ProcessorFactory.create(stage_config.name, **stage_config.kwargs))

        # build writer (only one)
        stage_config = config.writer
        writer = WriterFactory.create(stage_config.name, **stage_config.kwargs)

        return cls(source, extractor, processors, writer)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    ### EXECUTOR

    def execute(self) -> None:
        """
        Launch all stages.

        Blocks until all threads finish or an unrecoverable exception occurs.
        """
        def _runner(stage: Stage) -> None:
            try:
                stage.execute()
            except BaseException as e:
                logging.error(
                    "Stage crashed: %s", current_thread().name, exc_info=(type(e), e, e.__traceback__)
                )
                # signal everyone to stop
                self.close()
                raise

        # start each thread
        threads: list[Thread] = []
        for i, s in enumerate(self._stages):
            t = Thread(
                target=_runner, args=(s,), name=f"pipeline-stage-{i}-{type(s).name}", daemon=True
            )
            t.start()
            threads.append(t)

        # track the writer output for progress updates
        self.track()

        # join threads
        for thread in threads:
            thread.join()

    def track(self) -> None:
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
            refresh_per_second=10
        )

        with progress:
            for _ in progress.track(self._queues[-1], total=self._file_count, description="Processing"):
                pass

    def close(self) -> None:
        """Signal stages to stop, flush queues, and clean temporary resources."""
        # execute stop process on children
        if self._stages:
            for stage in self._stages:
                stage.close()

        # stop all queues
        for q in self._queues:
            q.close()

        # close scratch dir
        self._scratch.cleanup()
