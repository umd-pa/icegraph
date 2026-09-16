# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Iterator, Callable
from dataclasses import dataclass

from icegraph.common.plugins import PluginContext
from icegraph.common.record import GlobalAttributes, Attributes

__all__ = ["RecordDecoderContext"]


@dataclass(frozen=True)
class RecordDecoderContext(PluginContext):
    attrs:          Callable[[], Iterator[Attributes]]
    global_attrs:   GlobalAttributes

    # names and dtypes of the columns packed into a stored key, resolved by the
    # attribute decoder. empty for a key that recorded none
    columns:        Callable[[str], list[str]]
    dtypes:         Callable[[str], list[str]]
