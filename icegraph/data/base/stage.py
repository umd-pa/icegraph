# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

import time
import functools
from abc import abstractmethod, ABC
from typing import TYPE_CHECKING, Optional, Union, overload, Iterator
from queue import Queue
from pathlib import Path

from icegraph.config import IGConfig

if TYPE_CHECKING:
    from icegraph.data.pipeline import Pipeline, EnvelopeOrSentinel
else:
    Pipeline = None
    EnvelopeOrSentinel = None


class Stage(ABC):
    """
    Abstract base class for a single stage in the IceGraph data pipeline.

    A Stage consumes input items, processes them, and emits results to an
    output queue managed by a parent Pipeline.
    """

    def __init__(self):
        """Initialize an un-wired stage."""
        # must be set via configuration
        self._parent: Optional[Pipeline] = None
        self._out_queue_idx: Optional[int] = None
        self._in_iter: Optional[Union[Iterator[Pipeline.Envelope], Iterator[Path]]] = None

        # global config
        self._config: IGConfig = IGConfig.get()

    ### CONFIGURATIONS

    def set_parent(self, parent: Pipeline) -> None:
        """
        Attach this stage to a parent pipeline.

        Args:
            parent (Pipeline): The pipeline that manages this stage.
        """
        self._parent = parent

    def assign_queue(self, queue_idx: int) -> None:
        """
        Assign the output queue index for this stage.

        Args:
            queue_idx (int): Index of the output queue in the parent pipeline.
        """
        self._out_queue_idx = queue_idx

    @overload
    def set_in_iter(self, in_iter: Iterator[Pipeline.Envelope]) -> None: ...
    @overload
    def set_in_iter(self, in_iter: Iterator[Path]) -> None: ...

    def set_in_iter(self, in_iter: Union[Iterator[Pipeline.Envelope], Iterator[Path]]) -> None:
        """
        Set the input iterator for this stage.

        Args:
            in_iter (Iterator[Pipeline.Envelope] | Iterator[Path]):
                Input source producing envelopes or file paths.
        """
        self._in_iter = in_iter

    ### EXECUTORS

    def execute(self) -> None:
        """
        Consume inputs, process them, and emit results downstream.

        Raises:
            RuntimeError: If the stage is not wired with parent, queue, or input.
        """
        if self._parent is None or self._out_queue_idx is None or self._in_iter is None:
            raise RuntimeError(
                f"{type(self).__name__} not wired: set_parent(...), assign_queue(...), "
                f"and set_in_iter(...) must be called before execution."
            )
        try:
            for env in self._in_iter:
                if self._parent.stop.is_set():
                    break
                out = self._process(env)
                if out is not None:
                    self._out_q.put(out)  # blocks when full, backpressure
        finally:
            # Always signal end-of-stream downstream
            self._out_q.put(self._parent.SENTINEL)

    ### ACCESSORS

    @property
    def _out_q(self) -> Queue[EnvelopeOrSentinel]:
        """
        Return the output queue for this stage.

        Returns:
            Queue[EnvelopeOrSentinel]: The queue to emit processed results to.

        Raises:
            RuntimeError: If the stage is not wired with parent, queue, or input.
        """
        if self._parent is None or self._out_queue_idx is None or self._in_iter is None:
            raise RuntimeError(
                f"{type(self).__name__} not wired: set_parent(...), assign_queue(...), "
                f"and set_in_iter(...) must be called before execution."
            )
        return self._parent.queues[self._out_queue_idx]

    @staticmethod
    def profile():
        """
        Decorator factory, helper wrapper that measures runtime (ms) of a function that takes an Envelope.
        Stores it in env.metrics[stage_idx].
        """

        def decorator(func):
            @functools.wraps(func)
            def wrapper(self, *args, **kwargs):
                # run and profile func
                start = time.perf_counter()
                env: Pipeline.Envelope = func(self, *args, **kwargs)
                end = time.perf_counter()

                # calc elapsed time, save to metrics
                elapsed_ms = (end - start) * 1000
                stage_name = self.__class__.__name__
                env.metrics[stage_name] = elapsed_ms

                return env

            return wrapper

        return decorator

    ### STUBS

    @overload
    def _process(self, arg: Path) -> Optional[Pipeline.Envelope]: ...
    @overload
    def _process(self, arg: Pipeline.Envelope) -> Optional[Pipeline.Envelope]: ...

    @abstractmethod
    def _process(self, arg: Union[Path, Pipeline.Envelope]) -> Optional[Pipeline.Envelope]:
        """
        Transform a single input item into an output envelope.

        Args:
            arg (Path | Pipeline.Envelope): Input item.

        Returns:
            Optional[Pipeline.Envelope]: Processed envelope, or None to skip output.
        """
        ...

    @abstractmethod
    def bootstrap(self, infile: Path) -> Optional[Union[Pipeline.Envelope, Path]]:
        """
        Feed this stage from raw files instead of envelopes (for first-in-line stages).

        Args:
            infile (Path): Path to a raw input file.

        Returns:
            Optional[Pipeline.Envelope | Path]: An envelope or path to seed the pipeline,
            or None if no bootstrap item should be emitted.
        """
        ...