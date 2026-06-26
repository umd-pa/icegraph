# Copyright (c) 2025 University of Maryland and the IceCube Collaboration.
# Developed by Taylor St Jean

from __future__ import annotations

from typing import Iterator, Callable
from dataclasses import dataclass

from icegraph.common.plugins import PluginContext
from icegraph.common.record import GlobalAttributes, Attributes

__all__ = ["AttributeDecoderContext"]


@dataclass(frozen=True)
class AttributeDecoderContext(PluginContext):
    attrs:          Callable[[], Iterator[Attributes]]
    global_attrs:   GlobalAttributes
