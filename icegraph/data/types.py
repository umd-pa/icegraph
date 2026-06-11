# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Iterable
from dataclasses import dataclass
from pathlib import Path

from icegraph.common.plugins import PluginContext

from .shared.queue import IterableQueue
from .envelope import Envelope

__all__ = ["StageContext"]


@dataclass(frozen=True)
class StageContext(PluginContext):
    src: IterableQueue[Envelope] | Iterable[Path]
    dst: IterableQueue[Envelope] | None
    scratch: Path

    # ordering and index
    index: int
    total: int
