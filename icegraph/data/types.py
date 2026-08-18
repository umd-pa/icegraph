# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Generic, TypeVar
from dataclasses import dataclass
from pathlib import Path

from icegraph.common.plugins import PluginContext

from .shared.queue import IterableQueue
from .envelope import Envelope

if TYPE_CHECKING:
    from .envelope import Envelope

__all__ = ["StageContext"]


I = TypeVar("I", bound="Envelope | Path")


@dataclass(frozen=True)
class StageContext(PluginContext, Generic[I]):
    src: Iterable[I]
    dst: IterableQueue[Envelope] | None
    scratch: Path

    # ordering and index
    index: int
    total: int

    # if an outdir for persistent storage is required, specify here
    outdir: Path | None = None
