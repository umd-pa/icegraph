# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

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


class Operator(ABC):

    def __init__(self):
        # must be set via configuration
        self._parent: Optional[Pipeline] = None
        self._out_queue_idx: Optional[int] = None
        self._in_iter: Optional[Union[Iterator[Pipeline.Envelope], Iterator[Path]]] = None

        # global config
        self._config: IGConfig = IGConfig.get()

    ### CONFIGURATIONS

    def set_parent(self, parent: Pipeline) -> None:
        self._parent = parent

    def assign_queue(self, queue_idx: int) -> None:
        self._out_queue_idx = queue_idx

    @overload
    def set_in_iter(self, in_iter: Iterator[Pipeline.Envelope]) -> None: ...
    @overload
    def set_in_iter(self, in_iter: Iterator[Path]) -> None: ...
    def set_in_iter(self, in_iter: Union[Iterator[Pipeline.Envelope], Iterator[Path]]) -> None:
        self._in_iter = in_iter

    ### EXECUTORS

    def execute(self) -> None:
        """Consume input, process, and emit until the input ends or stop() is called."""
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
        if self._parent is None or self._out_queue_idx is None or self._in_iter is None:
            raise RuntimeError(
                f"{type(self).__name__} not wired: set_parent(...), assign_queue(...), "
                f"and set_in_iter(...) must be called before execution."
            )
        return self._parent.queues[self._out_queue_idx]

    ### STUBS

    @overload
    def _process(self, arg: Path) -> Optional[Pipeline.Envelope]: ...
    @overload
    def _process(self, arg: Pipeline.Envelope) -> Optional[Pipeline.Envelope]: ...
    @abstractmethod
    def _process(self, arg: Union[Path, Pipeline.Envelope]) -> Optional[Pipeline.Envelope]: ...

    @abstractmethod
    def bootstrap(self, infile: Path) -> Optional[Union[Pipeline.Envelope, Path]]: ...